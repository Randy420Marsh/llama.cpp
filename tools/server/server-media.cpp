#include "server-media.h"
#include "server-common.h"

#include "log.h"

#include <cpp-httplib/httplib.h>

#ifdef LLAMA_SUBPROCESS
#include <sheredom/subprocess.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <pthread.h>
#endif

using json = common_json;

//
// small helpers
//

static std::string fmt_hms(double t) {
    // 01:02:03.4
    if (t < 0) {
        t = 0;
    }
    const int h = (int)(t / 3600.0);
    const int m = (int)((t - h * 3600.0) / 60.0);
    const double s = t - h * 3600.0 - m * 60.0;
    char buf[48];
    snprintf(buf, sizeof(buf), "%02d:%02d:%04.1f", h, m, s);
    return buf;
}

static std::string fmt_num(double v, int prec = 1) {
    char buf[48];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

// seconds as a number, or "hh:mm:ss(.f)", "mm:ss(.f)", "ss(.f)", optionally with a trailing "s"
static double parse_time(const json & v, const std::string & what) {
    if (v.is_number()) {
        return v.get<double>();
    }
    if (!v.is_string()) {
        throw std::invalid_argument(what + " must be a number of seconds or a \"hh:mm:ss\" string");
    }
    std::string s = v.get<std::string>();
    if (!s.empty() && (s.back() == 's' || s.back() == 'S')) {
        s.pop_back();
    }
    double total = 0.0;
    size_t pos   = 0;
    int    parts = 0;
    while (true) {
        const size_t colon = s.find(':', pos);
        const std::string field = s.substr(pos, colon == std::string::npos ? std::string::npos : colon - pos);
        char * end = nullptr;
        const double x = strtod(field.c_str(), &end);
        if (field.empty() || end == field.c_str() || *end != '\0' || x < 0) {
            throw std::invalid_argument(what + " is not a valid time: \"" + v.get<std::string>() + "\"");
        }
        total = total * 60.0 + x;
        if (++parts > 3) {
            throw std::invalid_argument(what + " has too many ':' fields");
        }
        if (colon == std::string::npos) {
            break;
        }
        pos = colon + 1;
    }
    return total;
}

static std::vector<uint8_t> wav_from_s16le(const std::vector<uint8_t> & pcm, int sample_rate) {
    const uint32_t data_size = (uint32_t) pcm.size();
    std::vector<uint8_t> out(44 + pcm.size());
    auto put32 = [&](size_t off, uint32_t v) { memcpy(out.data() + off, &v, 4); };
    auto put16 = [&](size_t off, uint16_t v) { memcpy(out.data() + off, &v, 2); };
    memcpy(out.data() + 0, "RIFF", 4);
    put32(4, 36 + data_size);
    memcpy(out.data() + 8, "WAVEfmt ", 8);
    put32(16, 16);
    put16(20, 1);                          // PCM
    put16(22, 1);                          // mono
    put32(24, (uint32_t) sample_rate);
    put32(28, (uint32_t) sample_rate * 2); // byte rate
    put16(32, 2);                          // block align
    put16(34, 16);                         // bits per sample
    memcpy(out.data() + 36, "data", 4);
    put32(40, data_size);
    if (!pcm.empty()) {
        memcpy(out.data() + 44, pcm.data(), pcm.size());
    }
    return out;
}

//
// subprocess: run a program, optionally feeding stdin from memory, capturing stdout in memory
//

struct proc_result {
    int                  exit_code = -1;
    std::vector<uint8_t> out;
    std::string          err;
};

static proc_result run_capture(const std::vector<std::string> & args, const std::vector<uint8_t> * input) {
#ifndef LLAMA_SUBPROCESS
    GGML_UNUSED(args);
    GGML_UNUSED(input);
    throw std::runtime_error("video/audio processing needs a build with LLAMA_SUBPROCESS=ON");
#else
    std::vector<const char *> argv;
    for (const auto & a : args) {
        argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);

    subprocess_s proc = {};
    const int options = subprocess_option_search_user_path | subprocess_option_inherit_environment |
                        subprocess_option_no_window;
    if (subprocess_create(argv.data(), options, &proc) != 0) {
        throw std::runtime_error("failed to start '" + args[0] +
                                 "' - install ffmpeg or point --video-ffmpeg-dir at it");
    }

    proc_result res;
    std::thread feeder;
    if (input != nullptr) {
        // the buffer outlives the thread: it is joined below before returning
        feeder = std::thread([&proc, input]() {
#ifndef _WIN32
            // ffmpeg can exit before it reads all input; the write must fail with EPIPE, not kill the server with SIGPIPE
            sigset_t set;
            sigemptyset(&set);
            sigaddset(&set, SIGPIPE);
            pthread_sigmask(SIG_BLOCK, &set, nullptr);
#endif
            FILE * f = subprocess_stdin(&proc);
            if (f) {
                size_t done = 0;
                while (done < input->size()) {
                    const size_t n = fwrite(input->data() + done, 1, std::min<size_t>(input->size() - done, 1 << 20), f);
                    if (n == 0) {
                        break; // broken pipe: the child is done with its input
                    }
                    done += n;
                }
                fclose(f);
                proc.stdin_file = nullptr; // prevent a double close in subprocess_destroy
            }
        });
    } else if (proc.stdin_file) {
        fclose(proc.stdin_file);
        proc.stdin_file = nullptr;
    }

    std::thread err_reader([&proc, &res]() {
        FILE * f = subprocess_stderr(&proc);
        char buf[4096];
        size_t n;
        while (f && (n = fread(buf, 1, sizeof(buf), f)) > 0) {
            if (res.err.size() < 16384) {
                res.err.append(buf, n);
            }
        }
    });

    FILE * out = subprocess_stdout(&proc);
    std::vector<uint8_t> chunk(1 << 20);
    size_t n;
    while (out && (n = fread(chunk.data(), 1, chunk.size(), out)) > 0) {
        res.out.insert(res.out.end(), chunk.begin(), chunk.begin() + n);
    }

    // the feeder must finish before subprocess_join(), which closes stdin itself
#ifdef _WIN32
    // no SIGPIPE on windows: a blocked feeder only fails once the child exited and our copy of its stdin read end is closed
    WaitForSingleObject(proc.hProcess, INFINITE);
    if (proc.hStdInput) {
        CloseHandle(proc.hStdInput);
        proc.hStdInput = nullptr;
    }
#endif
    if (feeder.joinable()) {
        feeder.join(); // on posix the child's exit turns a blocked write into EPIPE
    }
    err_reader.join();
    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    while (!res.err.empty() && (res.err.back() == '\n' || res.err.back() == '\r')) {
        res.err.pop_back();
    }
    return res;
#endif
}

static std::string resolve_bin(const server_media_config & cfg, const char * name) {
    if (cfg.ffmpeg_bin_dir.empty()) {
        return name;
    }
    std::string out = cfg.ffmpeg_bin_dir;
    if (out.back() != '/' && out.back() != '\\') {
        out += '/';
    }
    out += name;
#ifdef _WIN32
    out += ".exe";
#endif
    return out;
}

//
// media sources
//

struct media_source {
    std::string          input;  // what ffmpeg opens: URL, path or "cache:pipe:0"
    std::vector<uint8_t> bytes;  // non-empty for in-memory sources, fed through stdin
    bool is_buffer() const { return !bytes.empty(); }
};

static media_source resolve_source(const std::string & value, const server_media_config & cfg, const std::string & what) {
    media_source src;
    if (value.empty()) {
        throw std::invalid_argument(what + " is empty");
    }
    if (string_starts_with(value, "http://") || string_starts_with(value, "https://")) {
        // streamed by ffmpeg with HTTP range requests: only the requested segments are downloaded
        src.input = value;
        return src;
    }
    if (string_starts_with(value, "file://")) {
        if (cfg.media_path.empty()) {
            throw std::invalid_argument("file:// URLs are not allowed unless --media-path is specified");
        }
        const std::string rel = value.substr(7);
        if (!fs_validate_filename(rel, true)) {
            throw std::invalid_argument("file path is not allowed: " + rel);
        }
        src.input = cfg.media_path + rel;
        return src;
    }
    if (string_starts_with(value, "data:")) {
        const size_t comma = value.find(',');
        if (comma == std::string::npos || value.substr(0, comma).find(";base64") == std::string::npos) {
            throw std::invalid_argument(what + " data URI must be base64 encoded");
        }
        src.bytes = base64_decode_media(value.substr(comma + 1));
    } else {
        src.bytes = base64_decode_media(value); // bare base64 (llama.cpp input_video / OpenAI input_audio)
    }
    if (src.bytes.empty()) {
        throw std::invalid_argument(what + " contains no data");
    }
    // cache: makes the pipe seekable, so MP4s with the index at the end can be read
    src.input = "cache:pipe:0";
    return src;
}

static void add_input_args(std::vector<std::string> & args, const media_source & src) {
    // plain containers only: playlist/concat/image-sequence demuxers open further URLs or files
    // named inside the media, which would get around --media-path
    args.insert(args.end(), {"-format_whitelist",
        "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,flv,mpegts,mpeg,asf,ogg,m4v,h264,hevc,ivf,obu,"
        "wav,w64,mp3,aac,flac,aiff,caf,amr,wv"});
    if (src.is_buffer()) {
        args.insert(args.end(), {"-protocol_whitelist", "cache,pipe", "-read_ahead_limit", "-1"});
    } else if (string_starts_with(src.input, "http://") || string_starts_with(src.input, "https://")) {
        args.insert(args.end(), {"-protocol_whitelist", "http,https,tcp,tls"});
    } else {
        args.insert(args.end(), {"-protocol_whitelist", "file"});
    }
    args.insert(args.end(), {"-i", src.input});
}

struct media_probe {
    double duration   = 0.0;
    int    width      = 0;
    int    height     = 0;
    bool   has_video  = false;
    bool   has_audio  = false;
};

static media_probe probe_media(const media_source & src, const server_media_config & cfg) {
    std::vector<std::string> args = {
        resolve_bin(cfg, "ffprobe"), "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,width,height:stream_side_data=rotation",
        "-of", "json",
    };
    add_input_args(args, src);
    const proc_result r = run_capture(args, src.is_buffer() ? &src.bytes : nullptr);
    if (r.exit_code != 0 || r.out.empty()) {
        throw std::invalid_argument("cannot read media (ffprobe): " + (r.err.empty() ? std::string("unknown format") : r.err));
    }
    const json info = json::parse(std::string(r.out.begin(), r.out.end()));
    media_probe p;
    if (info.contains("format") && info.at("format").contains("duration")) {
        p.duration = atof(info.at("format").at("duration").get<std::string>().c_str());
    }
    if (info.contains("streams")) {
        for (const auto & s : info.at("streams")) {
            const std::string type = s.value("codec_type", std::string());
            if (type == "video" && !p.has_video) {
                p.has_video = true;
                p.width     = s.value("width", 0);
                p.height    = s.value("height", 0);
                if (s.contains("side_data_list")) {
                    for (const auto & sd : s.at("side_data_list")) {
                        const int rot = std::abs((int) sd.value("rotation", 0.0));
                        if (rot == 90 || rot == 270) {
                            std::swap(p.width, p.height); // ffmpeg autorotates, so plan for the displayed size
                        }
                    }
                }
            } else if (type == "audio") {
                p.has_audio = true;
            }
        }
    }
    return p;
}

//
// options shared by video and audio parts
//

struct media_segment {
    double start = 0.0;
    double end   = 0.0;
    double length() const { return end - start; }
};

struct media_options {
    std::string              source;
    std::vector<media_segment> segments;
    bool                     explicit_range = false;
    double                   fps        = 0.0;
    int                      detail_tokens = 0;
    std::string              detail_name;
    int                      max_tokens = 0;
    int                      max_frames = 0;
    std::string              audio_mode = "auto"; // auto | native | transcript | both | none
    std::string              language;
};

static int detail_to_tokens(const std::string & name) {
    // tokens per frame at 32x32 px per token; measured: 20px text on a 1080p frame is readable from "standard" up, small UI text needs "max"
    if (name == "low")      return 256;
    if (name == "standard" || name == "auto" || name == "medium") return 576;
    if (name == "high")     return 1024;
    if (name == "max")      return 2048;
    throw std::invalid_argument("detail must be one of low, standard, high, max (or a number of tokens per frame)");
}

static media_options parse_options(const json & part, const std::string & field, const server_media_config & cfg) {
    media_options o;
    o.fps           = cfg.video_fps;
    o.detail_name   = cfg.video_detail;
    o.detail_tokens = detail_to_tokens(cfg.video_detail);
    o.max_tokens    = cfg.video_max_tokens;
    o.max_frames    = cfg.video_max_frames;
    o.language      = cfg.asr_language;

    if (!part.contains(field)) {
        throw std::invalid_argument("'" + field + "' content part must contain '" + field + "'");
    }
    const json & v = part.at(field);
    if (v.is_string()) {
        o.source = v.get<std::string>(); // {"video_url": "https://..."} shorthand
        return o;
    }
    if (!v.is_object()) {
        throw std::invalid_argument("'" + field + "' must be a URL string or an object");
    }
    // OpenAI shape uses "url", llama.cpp input_video/input_audio use "data"
    if (v.contains("data")) {
        o.source = v.at("data").get<std::string>();
    } else if (v.contains("url")) {
        o.source = v.at("url").get<std::string>();
    } else {
        throw std::invalid_argument("'" + field + "' must contain 'url' or 'data'");
    }

    if (v.contains("segments")) {
        const json & segs = v.at("segments");
        if (!segs.is_array() || segs.size() == 0) {
            throw std::invalid_argument(field + ".segments must be a non-empty array of {start, end}");
        }
        for (const auto & s : segs) {
            media_segment seg;
            seg.start = s.contains("start") ? parse_time(s.at("start"), field + ".segments[].start") : 0.0;
            if (s.contains("end")) {
                seg.end = parse_time(s.at("end"), field + ".segments[].end");
            } else if (s.contains("duration")) {
                seg.end = seg.start + parse_time(s.at("duration"), field + ".segments[].duration");
            } else {
                seg.end = -1.0; // to the end of the media
            }
            o.segments.push_back(seg);
        }
        o.explicit_range = true;
    } else if (v.contains("start") || v.contains("end") || v.contains("duration")) {
        media_segment seg;
        seg.start = v.contains("start") ? parse_time(v.at("start"), field + ".start") : 0.0;
        if (v.contains("end")) {
            seg.end = parse_time(v.at("end"), field + ".end");
        } else if (v.contains("duration")) {
            seg.end = seg.start + parse_time(v.at("duration"), field + ".duration");
        } else {
            seg.end = -1.0;
        }
        o.segments.push_back(seg);
        o.explicit_range = true;
    }
    if (o.segments.size() > 256) {
        throw std::invalid_argument(field + ".segments: at most 256 segments per part");
    }

    if (v.contains("fps")) {
        o.fps = v.at("fps").get<double>();
        if (!(o.fps > 0.0)) {
            throw std::invalid_argument(field + ".fps must be positive");
        }
        // --video-fps is also the ceiling: the model helper never samples above it
        o.fps = std::min(o.fps, (double) cfg.video_fps);
    }
    if (v.contains("detail")) {
        const json & d = v.at("detail");
        if (d.is_number()) {
            o.detail_tokens = std::max(16, d.get<int>());
            o.detail_name   = std::to_string(o.detail_tokens) + " tokens";
        } else {
            o.detail_name   = d.get<std::string>();
            o.detail_tokens = detail_to_tokens(o.detail_name);
        }
    }
    // a request may lower the server limits, never raise them
    if (v.contains("max_tokens")) {
        o.max_tokens = std::clamp(v.at("max_tokens").get<int>(), 64, cfg.video_max_tokens);
    }
    if (v.contains("max_frames")) {
        o.max_frames = std::clamp(v.at("max_frames").get<int>(), 1, cfg.video_max_frames);
    }
    if (v.contains("audio")) {
        const json & a = v.at("audio");
        o.audio_mode = a.is_boolean() ? (a.get<bool>() ? "auto" : "none") : a.get<std::string>();
    } else if (v.contains("transcribe")) {
        o.audio_mode = v.at("transcribe").get<bool>() ? "transcript" : "none";
    }
    if (o.audio_mode != "auto" && o.audio_mode != "native" && o.audio_mode != "transcript" &&
        o.audio_mode != "both" && o.audio_mode != "none") {
        throw std::invalid_argument(field + ".audio must be one of auto, native, transcript, both, none");
    }
    if (v.contains("language")) {
        o.language = v.at("language").get<std::string>();
    }
    return o;
}

static void finalize_segments(media_options & o, double duration, const std::string & what) {
    if (o.segments.empty()) {
        if (!(duration > 0.0)) {
            throw std::invalid_argument(what + ": cannot determine the media duration; give an explicit start/end");
        }
        o.segments.push_back({0.0, duration});
        return;
    }
    for (auto & s : o.segments) {
        if (s.end < 0.0) {
            if (!(duration > 0.0)) {
                throw std::invalid_argument(what + ": segment without end, and the media duration is unknown");
            }
            s.end = duration;
        }
        if (duration > 0.0) {
            if (s.start >= duration) {
                throw std::invalid_argument(what + ": segment start " + fmt_hms(s.start) +
                                            " is beyond the end of the media (" + fmt_hms(duration) + ")");
            }
            s.end = std::min(s.end, duration);
        }
        if (!(s.end > s.start)) {
            throw std::invalid_argument(what + ": segment end must be after its start (" + fmt_hms(s.start) + ")");
        }
    }
}

//
// audio: extraction, native pass-through, speech-to-text
//

static std::vector<uint8_t> extract_audio_wav(const media_source & src, const media_segment & seg, bool whole,
                                              const server_media_config & cfg) {
    std::vector<std::string> args = {resolve_bin(cfg, "ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error"};
    if (!whole) {
        args.insert(args.end(), {"-ss", fmt_num(seg.start, 3)});
    }
    add_input_args(args, src);
    if (!whole) {
        args.insert(args.end(), {"-t", fmt_num(seg.length(), 3)});
    }
    args.insert(args.end(), {"-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
                             "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"});
    const proc_result r = run_capture(args, src.is_buffer() ? &src.bytes : nullptr);
    if (r.exit_code != 0) {
        throw std::invalid_argument("cannot decode audio: " + r.err);
    }
    return wav_from_s16le(r.out, 16000);
}

struct asr_segment {
    double      start = 0.0;
    double      end   = 0.0;
    std::string text;
};

static std::vector<asr_segment> transcribe(const std::vector<uint8_t> & wav, const std::string & language,
                                           const server_media_config & cfg) {
    // split "http(s)://host[:port][/path]"; the path defaults to the OpenAI transcription route
    std::string url = cfg.asr_url;
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw std::runtime_error("--asr-url must start with http:// or https://");
    }
    const size_t path_start = url.find('/', scheme_end + 3);
    std::string base = path_start == std::string::npos ? url : url.substr(0, path_start);
    std::string path = path_start == std::string::npos ? "" : url.substr(path_start);
    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
    if (path.empty()) {
        path = "/v1/audio/transcriptions";
    } else if (path.size() >= 3 && path.compare(path.size() - 3, 3, "/v1") == 0) {
        path += "/audio/transcriptions";
    }

    httplib::Client cli(base);
    cli.set_connection_timeout(10, 0);
    cli.set_write_timeout(600, 0);
    cli.set_read_timeout(3600, 0); // an hour of audio takes a minute or two on a GPU

    httplib::UploadFormDataItems items = {
        {"file",            std::string(wav.begin(), wav.end()), "audio.wav", "audio/wav"},
        {"response_format", "verbose_json",                       "",          ""},
    };
    if (!cfg.asr_model.empty()) {
        items.push_back({"model", cfg.asr_model, "", ""});
    }
    if (!language.empty()) {
        items.push_back({"language", language, "", ""});
    }
    auto res = cli.Post(path, items);
    if (!res) {
        throw std::runtime_error("speech-to-text server at " + cfg.asr_url + " is not reachable: " +
                                 httplib::to_string(res.error()));
    }
    if (res->status != 200) {
        throw std::runtime_error("speech-to-text server returned HTTP " + std::to_string(res->status) + ": " +
                                 res->body.substr(0, 300));
    }
    const json body = json::parse(res->body);
    std::vector<asr_segment> out;
    if (body.contains("segments")) {
        for (const auto & s : body.at("segments")) {
            asr_segment seg;
            seg.start = s.value("start", 0.0);
            seg.end   = s.value("end", 0.0);
            seg.text  = s.value("text", std::string());
            const size_t b = seg.text.find_first_not_of(" \t\r\n");
            const size_t e = seg.text.find_last_not_of(" \t\r\n");
            seg.text = b == std::string::npos ? "" : seg.text.substr(b, e - b + 1);
            if (!seg.text.empty()) {
                out.push_back(std::move(seg));
            }
        }
    } else if (body.contains("text")) {
        // plain json response: one untimed segment
        std::string text = body.at("text").get<std::string>();
        if (!text.empty()) {
            out.push_back({0.0, 0.0, text});
        }
    }
    return out;
}

static std::string format_transcript(const std::vector<asr_segment> & segs, double offset, const std::string & title) {
    std::string out = title + " (speech-to-text; times are seconds in the source):\n";
    if (segs.empty()) {
        return out + "(no speech detected)";
    }
    for (const auto & s : segs) {
        if (s.end > s.start) {
            out += "<" + fmt_num(offset + s.start) + " - " + fmt_num(offset + s.end) + " seconds> " + s.text + "\n";
        } else {
            out += s.text + "\n";
        }
    }
    out.pop_back();
    return out;
}

// auto: a video's audio goes to the timed transcript when --asr-url is set (more exact than Gemma4 native audio on compressed tracks); a standalone audio part stays native when the model has an audio encoder
static std::string resolve_audio_mode(const std::string & requested, double seconds, const server_media_config & cfg,
                                      const std::string & what, bool with_video) {
    std::string mode = requested;
    if (mode == "auto") {
        const bool native_ok = cfg.audio_native && seconds <= cfg.audio_native_max_s;
        if (with_video && !cfg.asr_url.empty()) {
            mode = "transcript";
        } else if (native_ok) {
            mode = "native";
        } else if (!cfg.asr_url.empty()) {
            mode = "transcript";
        } else {
            mode = "none";
        }
    }
    if ((mode == "native" || mode == "both") && !cfg.audio_native) {
        throw std::invalid_argument(what + ": this model has no audio encoder; use audio \"transcript\" "
                                    "(start the server with --asr-url) or \"none\"");
    }
    if ((mode == "transcript" || mode == "both") && cfg.asr_url.empty()) {
        throw std::invalid_argument(what + ": transcripts need a speech-to-text server; start llama-server with --asr-url");
    }
    return mode;
}

static json text_part(const std::string & text) {
    json p = json::object();
    p["type"] = "text";
    p["text"] = text;
    return p;
}

static json marker_part() {
    json p = json::object();
    p["type"] = "media_marker";
    p["text"] = get_media_marker();
    return p;
}

//
// video
//

json server_media_video_part(const json & part, const server_media_config & cfg, std::vector<std::vector<uint8_t>> & out_files) {
    const std::string type  = part.value("type", std::string());
    const std::string field = type == "input_video" ? "input_video" : "video_url";
    if (!cfg.video_supported) {
        throw std::runtime_error("video input is not supported - hint: this needs an mmproj with a vision encoder "
                                 "and a build with MTMD_VIDEO (ffmpeg)");
    }
    media_options o = parse_options(part, field, cfg);
    const media_source src = resolve_source(o.source, cfg, field);
    const media_probe  p   = probe_media(src, cfg);
    if (!p.has_video || p.width <= 0 || p.height <= 0) {
        throw std::invalid_argument(field + ": the media has no video stream");
    }
    finalize_segments(o, p.duration, field);

    // frame size: aspect preserved, area from the detail level, never upscaled, multiples of 32
    const double area_target = (double) o.detail_tokens * 1024.0;
    const double scale       = std::min(1.0, std::sqrt(area_target / ((double) p.width * p.height)));
    const int    w = std::max(32, (int) std::lround(p.width  * scale / 32.0) * 32);
    const int    h = std::max(32, (int) std::lround(p.height * scale / 32.0) * 32);

    double frame_tokens = cfg.image_tokens_per_pixel * w * h;
    if (cfg.image_tokens_cap > 0) {
        frame_tokens = std::min(frame_tokens, (double) cfg.image_tokens_cap);
    }
    frame_tokens = std::max(1.0, frame_tokens / std::max(1, cfg.n_temporal_merge));

    // fps: requested/default, lowered to fit max_frames and max_tokens; never below the floor
    double total_len = 0.0;
    for (const auto & s : o.segments) {
        total_len += s.length();
    }
    double fps = o.fps;
    fps = std::min(fps, o.max_frames / total_len);
    fps = std::min(fps, o.max_tokens / (frame_tokens * total_len));
    if (fps < cfg.video_min_fps) {
        const double max_len = std::min(o.max_frames / (double) cfg.video_min_fps,
                                        o.max_tokens / (frame_tokens * cfg.video_min_fps));
        throw std::invalid_argument(
            field + ": " + fmt_num(total_len, 0) + " s of video does not fit: at detail \"" + o.detail_name + "\" (" +
            std::to_string(w) + "x" + std::to_string(h) + ", ~" + fmt_num(frame_tokens, 0) + " tokens per frame) and " +
            "max_tokens=" + std::to_string(o.max_tokens) + ", max_frames=" + std::to_string(o.max_frames) +
            " even " + fmt_num(cfg.video_min_fps, 2) + " fps allows at most " + fmt_num(max_len, 0) + " s. " +
            "Use shorter segments (start/end or segments[]), a lower detail, a higher max_tokens, or split the video "
            "into chunks and summarize them one by one.");
    }

    SRV_INF("video: %s, %d segment(s), %.1f s total, %dx%d -> %dx%d, %.3f fps (requested %.2f), ~%.0f tokens/frame, est. %.0f tokens\n",
            src.is_buffer() ? "in-memory" : src.input.c_str(), (int) o.segments.size(), total_len,
            p.width, p.height, w, h, fps, o.fps, frame_tokens, frame_tokens * fps * total_len);

    json out = json::array();
    const bool several = o.segments.size() > 1;
    for (size_t i = 0; i < o.segments.size(); i++) {
        const media_segment & seg = o.segments[i];
        const bool whole = !o.explicit_range;

        if (o.explicit_range) {
            std::string head = several ? "Video segment " + std::to_string(i + 1) + " of " + std::to_string(o.segments.size())
                                       : std::string("Video segment");
            out.push_back(text_part(head + ", source time " + fmt_hms(seg.start) + " to " + fmt_hms(seg.end) + ":"));
        }

        // in-memory FFV1 clip: exact (lossless) frames at the planned fps and size, timestamps shifted to the source time
        std::vector<std::string> args = {resolve_bin(cfg, "ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error"};
        if (!whole) {
            args.insert(args.end(), {"-ss", fmt_num(seg.start, 3)});
        }
        add_input_args(args, src);
        if (!whole) {
            args.insert(args.end(), {"-t", fmt_num(seg.length(), 3)});
        }
        args.insert(args.end(), {
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", "fps=" + fmt_num(fps, 6) + ",scale=" + std::to_string(w) + ":" + std::to_string(h) +
                   ":flags=lanczos,format=yuv420p",
            "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "4",
            "-output_ts_offset", fmt_num(seg.start, 3),
            "-f", "matroska", "pipe:1",
        });
        proc_result r = run_capture(args, src.is_buffer() ? &src.bytes : nullptr);
        if (r.exit_code != 0 || r.out.empty()) {
            throw std::invalid_argument(field + ": cannot decode video segment " + fmt_hms(seg.start) + " - " +
                                        fmt_hms(seg.end) + ": " + (r.err.empty() ? std::string("no frames") : r.err));
        }
        SRV_INF("video: segment %s - %s -> %.1f MiB clip\n", fmt_hms(seg.start).c_str(), fmt_hms(seg.end).c_str(),
                r.out.size() / 1048576.0);

        const std::string mode = p.has_audio ? resolve_audio_mode(o.audio_mode, seg.length(), cfg, field, true) : "none";
        std::vector<uint8_t> wav;
        if (mode != "none") {
            wav = extract_audio_wav(src, seg, whole, cfg);
        }
        // native audio goes before the frames: after them, Gemma4 made up speech for every frame
        if (mode == "native" || mode == "both") {
            out.push_back(text_part("Audio of this video segment (starts at source time " + fmt_hms(seg.start) + "):"));
            out_files.push_back(wav);
            out.push_back(marker_part());
        }
        out_files.push_back(std::move(r.out));
        out.push_back(marker_part());

        if (!p.has_audio && (o.audio_mode == "native" || o.audio_mode == "transcript" || o.audio_mode == "both")) {
            out.push_back(text_part("(the video has no audio track)"));
        }
        if (mode == "transcript" || mode == "both") {
            const auto segs = transcribe(wav, o.language, cfg);
            SRV_INF("video: transcript of %s - %s has %zu segments\n", fmt_hms(seg.start).c_str(), fmt_hms(seg.end).c_str(), segs.size());
            out.push_back(text_part(format_transcript(segs, seg.start, "Audio transcript of this video segment")));
        }
    }
    return out;
}

//
// audio
//

json server_media_audio_part(const json & part, const server_media_config & cfg, std::vector<std::vector<uint8_t>> & out_files) {
    const std::string type  = part.value("type", std::string());
    const std::string field = type == "audio_url" ? "audio_url" : "input_audio";
    media_options o = parse_options(part, field, cfg);
    const media_source src = resolve_source(o.source, cfg, field);
    const bool whole = !o.explicit_range;

    // whole in-memory clip for an audio-capable model: pass it as-is, so this case works without ffmpeg (mtmd decodes wav/mp3/flac)
    if (whole && src.is_buffer() && cfg.audio_native && (o.audio_mode == "auto" || o.audio_mode == "native")) {
        out_files.push_back(src.bytes);
        json out = json::array();
        out.push_back(marker_part());
        return out;
    }

    const media_probe p = probe_media(src, cfg);
    if (!p.has_audio) {
        throw std::invalid_argument(field + ": the media has no audio stream");
    }
    finalize_segments(o, p.duration, field);

    json out = json::array();
    for (size_t i = 0; i < o.segments.size(); i++) {
        const media_segment & seg = o.segments[i];
        const std::string mode = resolve_audio_mode(o.audio_mode, seg.length(), cfg, field, false);
        if (mode == "none") {
            throw std::invalid_argument(field + ": this model has no audio encoder and no speech-to-text server is "
                                        "configured - start llama-server with --asr-url");
        }
        if (o.explicit_range) {
            out.push_back(text_part("Audio segment, source time " + fmt_hms(seg.start) + " to " + fmt_hms(seg.end) + ":"));
        }
        std::vector<uint8_t> wav = extract_audio_wav(src, seg, whole, cfg);
        if (mode == "transcript" || mode == "both") {
            const auto segs = transcribe(wav, o.language, cfg);
            SRV_INF("audio: transcript of %s - %s has %zu segments\n", fmt_hms(seg.start).c_str(), fmt_hms(seg.end).c_str(), segs.size());
            out.push_back(text_part(format_transcript(segs, seg.start, "Audio transcript")));
        }
        if (mode == "native" || mode == "both") {
            out_files.push_back(std::move(wav));
            out.push_back(marker_part());
        }
    }
    return out;
}
