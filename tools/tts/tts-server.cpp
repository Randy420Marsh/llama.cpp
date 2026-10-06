// llama-tts-server: keeps a speech model (backbone + mmproj with audio generation)
// resident and serves it over HTTP, so a request costs generation time only.
// llama-tts reloads the model on every run, which is fine for a test and far
// too slow for reading a chat reply aloud sentence by sentence.
//
//   llama-tts-server -m Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
//       -mm mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf \
//       --tts-voices-dir C:\AI\tts-models\voices --port 8181
//
// Zero-shot voice cloning needs no training: every audio file in the voices
// directory is a voice, used as the speaker reference (`--tts-speaker-file` of
// llama-tts). Two layouts are recognised:
//   <dir>/<name>.wav|mp3|flac
//   <dir>/<name>/reference.wav|mp3|flac   (+ optional voice.json {"language": "en"})
//
// endpoints
//   GET  /health            {"status":"ok"}
//   GET  /v1/models         model id, generator, sample rate, languages
//   GET  /v1/audio/voices   the voices found in --tts-voices-dir
//   POST /v1/audio/speech   {"input", "voice", "language", "seed", "response_format": "wav"|"pcm"}
//                           -> audio/wav (16-bit mono), or raw s16le for "pcm"

#ifdef _WIN32
// before any header can pull in windows.h: keeps winsock.h out, cpp-httplib needs winsock2.h
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

// Qwen3-TTS language codes; other generators ignore the language
static const char * k_languages[] = { "en", "zh", "de", "it", "pt", "es", "ja", "ko", "fr", "ru" };

// longer inputs are split at sentence ends and generated piece by piece: a long
// single generation drifts, and one piece bounds the work done per decode loop
static constexpr size_t k_chunk_chars  = 360;
static constexpr int    k_gap_ms       = 140;   // silence between pieces
static constexpr size_t k_max_chars    = 20000; // per request

struct server_opts {
    std::string host       = "127.0.0.1";
    int         port       = 8181;
    std::string voices_dir;
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m backbone.gguf -mm mmproj.gguf --tts-voices-dir voices --port 8181", argv[0]);
    LOG("\n    %s -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF --port 8181\n", argv[0]);
    LOG("\nserver options (in addition to the llama-tts options above):\n");
    LOG("  --host HOST            listen address (default: 127.0.0.1)\n");
    LOG("  --port PORT            listen port (default: 8181)\n");
    LOG("  --tts-voices-dir DIR   folder of reference clips, one voice per file or sub-folder\n");
    LOG("\n");
}

// The server flags are not llama-tts flags, so they are taken out before the
// common parser sees the command line. On Windows the arguments are re-read as
// UTF-8 first: the parser only repairs an argv that matches the process command
// line, and a filtered one would otherwise keep the ANSI code page.
static std::vector<std::string> utf8_args(int argc, char ** argv) {
    std::vector<std::string> out;
#ifdef _WIN32
    int wargc = 0;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv && wargc == argc) {
        for (int i = 0; i < wargc; i++) {
            const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(n > 0 ? (size_t) n - 1 : 0, '\0');
            if (n > 1) {
                WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
            }
            out.push_back(std::move(s));
        }
        LocalFree(wargv);
        return out;
    }
    if (wargv) {
        LocalFree(wargv);
    }
#endif
    for (int i = 0; i < argc; i++) {
        out.emplace_back(argv[i]);
    }
    return out;
}

static bool take_server_opts(std::vector<std::string> & args, server_opts & opts) {
    std::vector<std::string> rest;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string & a = args[i];
        const bool has_val = i + 1 < args.size();
        if (a == "--host" || a == "--port" || a == "--tts-voices-dir") {
            if (!has_val) {
                fprintf(stderr, "error: %s needs a value\n", a.c_str());
                return false;
            }
            const std::string & v = args[++i];
            if (a == "--host") {
                opts.host = v;
            } else if (a == "--port") {
                opts.port = std::atoi(v.c_str());
            } else {
                opts.voices_dir = v;
            }
            continue;
        }
        rest.push_back(a);
    }
    args = std::move(rest);
    return opts.port > 0 && opts.port < 65536;
}

static fs::path u8_path(const std::string & s) {
#if defined(__cpp_char8_t)
    return fs::path(std::u8string(s.begin(), s.end()));
#else
    return fs::u8path(s);
#endif
}

static std::string path_u8(const fs::path & p) {
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

static bool is_audio_ext(const fs::path & p) {
    std::string ext = path_u8(p.extension());
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return ext == ".wav" || ext == ".mp3" || ext == ".flac";
}

struct voice_entry {
    std::string name;
    fs::path    file;
    std::string language; // from <name>/voice.json, may be empty
    uintmax_t   bytes = 0;
};

// one voice per audio file, or per sub-folder holding reference.<ext>
static std::vector<voice_entry> scan_voices(const std::string & dir) {
    std::vector<voice_entry> out;
    if (dir.empty()) {
        return out;
    }
    std::error_code ec;
    const fs::path root = u8_path(dir);
    if (!fs::is_directory(root, ec)) {
        return out;
    }
    for (const auto & ent : fs::directory_iterator(root, ec)) {
        if (ent.is_regular_file(ec) && is_audio_ext(ent.path())) {
            voice_entry v;
            v.name  = path_u8(ent.path().stem());
            v.file  = ent.path();
            v.bytes = ent.file_size(ec);
            out.push_back(v);
        } else if (ent.is_directory(ec)) {
            for (const char * ext : { ".wav", ".flac", ".mp3" }) {
                const fs::path ref = ent.path() / (std::string("reference") + ext);
                if (fs::is_regular_file(ref, ec)) {
                    voice_entry v;
                    v.name  = path_u8(ent.path().filename());
                    v.file  = ref;
                    v.bytes = fs::file_size(ref, ec);
                    std::ifstream meta(ent.path() / "voice.json");
                    if (meta) {
                        const json j = json::parse(meta, nullptr, false);
                        if (j.is_object() && j.contains("language") && j["language"].is_string()) {
                            v.language = j["language"].get<std::string>();
                        }
                    }
                    out.push_back(v);
                    break;
                }
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const voice_entry & a, const voice_entry & b) { return a.name < b.name; });
    return out;
}

// split at sentence ends (then commas, then spaces) into pieces of at most `limit` bytes
static std::vector<std::string> split_text(const std::string & text, size_t limit) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < text.size()) {
        while (start < text.size() && (text[start] == ' ' || text[start] == '\n' || text[start] == '\r' || text[start] == '\t')) {
            start++;
        }
        if (start >= text.size()) {
            break;
        }
        if (text.size() - start <= limit) {
            out.push_back(text.substr(start));
            break;
        }
        const std::string window = text.substr(start, limit);
        size_t cut = std::string::npos;
        for (const char * marks : { ".!?\n", ",;:", " " }) {
            const size_t p = window.find_last_of(marks);
            if (p != std::string::npos && p > limit / 3) {
                cut = p + 1;
                break;
            }
        }
        if (cut == std::string::npos) {
            cut = limit;
            // never split inside a UTF-8 sequence
            while (cut > 0 && (((unsigned char) text[start + cut]) & 0xC0) == 0x80) {
                cut--;
            }
            if (cut == 0) {
                cut = limit;
            }
        }
        out.push_back(text.substr(start, cut));
        start += cut;
    }
    return out;
}

static void append_wav_header(std::string & buf, uint32_t n_samples, int32_t rate) {
    const uint32_t data_sz = n_samples * 2;
    const uint32_t riff_sz = 36 + data_sz;
    const uint32_t fmt_sz = 16, byte_rate = (uint32_t) rate * 2, rate32 = (uint32_t) rate;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    auto put = [&](const void * p, size_t n) { buf.append((const char *) p, n); };
    put("RIFF", 4); put(&riff_sz, 4); put("WAVE", 4);
    put("fmt ", 4); put(&fmt_sz, 4);
    put(&fmt, 2); put(&ch, 2); put(&rate32, 4); put(&byte_rate, 4); put(&align, 2); put(&bits, 2);
    put("data", 4); put(&data_sz, 4);
}

static void append_pcm16(std::string & buf, const std::vector<float> & pcm) {
    const size_t off = buf.size();
    buf.resize(off + pcm.size() * 2);
    int16_t * dst = (int16_t *) (buf.data() + off);
    for (size_t i = 0; i < pcm.size(); i++) {
        dst[i] = (int16_t) (std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f);
    }
}

struct tts_engine {
    common_params    params;
    llama_model    * model = nullptr;
    llama_context  * lctx  = nullptr;
    // members are destroyed in reverse order: the mtmd context (and the speaker
    // bitmaps below) must go before the model they were created against
    common_init_result_ptr init;
    mtmd::context_ptr mctx;
    mtmd_gen_audio_info info{};
    std::string model_id;
    std::string voices_dir;
    std::mutex  mtx; // one generation at a time: there is one context

    struct speaker_cache {
        fs::file_time_type mtime{};
        uintmax_t          bytes = 0;
        mtmd::bitmap_ptr   bitmap;
    };
    std::map<std::string, speaker_cache> speakers; // by file path

    // decoded speaker reference, re-read only when the file changes
    mtmd_bitmap * speaker(const fs::path & file, std::string & err) {
        std::error_code ec;
        const auto mtime = fs::last_write_time(file, ec);
        const auto bytes = fs::file_size(file, ec);
        if (ec) {
            err = "cannot read " + path_u8(file);
            return nullptr;
        }
        auto & c = speakers[path_u8(file)];
        if (c.bitmap && c.mtime == mtime && c.bytes == bytes) {
            return c.bitmap.get();
        }
        std::ifstream f(file, std::ios::binary);
        std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (data.empty()) {
            err = "empty speaker file " + path_u8(file);
            return nullptr;
        }
        auto w = mtmd_helper_bitmap_init_from_buf(mctx.get(), data.data(), data.size(), false, mtmd_helper_init_opt_default());
        if (!w.bitmap) {
            err = "could not decode speaker file " + path_u8(file) + " (wav, mp3 or flac)";
            return nullptr;
        }
        c.bitmap.reset(w.bitmap);
        c.mtime = mtime;
        c.bytes = bytes;
        return c.bitmap.get();
    }

    // one piece of text -> PCM, appended to `pcm`; returns frames generated, -1 on error
    int generate(const std::string & text, mtmd_bitmap * spk, const std::string & lang, uint32_t seed,
                 std::vector<float> & pcm, std::string & err) {
        llama_memory_clear(llama_get_memory(lctx), true);

        common_params_sampling sparams = params.sampling;
        sparams.seed = seed;
        common_sampler * smpl = common_sampler_init(model, sparams);
        if (!smpl) {
            err = "sampler init failed";
            return -1;
        }

        mtmd_helper::gen_audio gen(lctx, mctx.get());
        mtmd_helper_gen_audio_inp inp{};
        inp.seq_id      = 0;
        inp.prompt      = text.c_str();
        inp.prompt_len  = text.size();
        inp.speaker_ref = spk;
        inp.lang        = lang.c_str();
        inp.top_k       = params.sampling.top_k;
        inp.top_p       = params.sampling.top_p;
        inp.seed        = seed;
        inp.out_type    = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

        auto fail = [&](const std::string & why) {
            err = why;
            common_sampler_free(smpl);
            return -1;
        };

        if (gen.set_input(&inp) != 0) {
            return fail("set_input failed (unsupported language or bad speaker file?)");
        }
        for (;;) {
            const int32_t ret = gen.step_prompt(params.n_batch);
            if (ret < 0) {
                return fail("prompt processing failed");
            }
            if (ret == 0) {
                break;
            }
        }

        // ~12.5 frames per second of speech; budget generously from the text length,
        // but never past the context (the prompt holds at most one position per byte)
        const int ctx_room = (int) llama_n_ctx(lctx) - (int) text.size() - 64;
        const int budget   = std::max(1, std::min(ctx_room,
            params.n_predict > 0 ? params.n_predict : (int) std::min<size_t>(4000, 120 + text.size() * 2)));

        auto sample = [&]() {
            const llama_token t = common_sampler_sample(smpl, lctx, -1);
            common_sampler_accept(smpl, t, true);
            return t;
        };
        llama_token sampled = sample();
        const float * h_state = llama_get_embeddings_ith(lctx, -1);
        int  n_frames = 0;
        bool stop     = false;
        while (!stop && n_frames < budget) {
            const float * h_next = nullptr;
            if (gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
                return fail("generation failed at frame " + std::to_string(n_frames));
            }
            if (!h_next) {
                break;
            }
            n_frames++;
            h_state = h_next;
            sampled = sample();
        }
        if (n_frames >= budget) {
            LOG_WRN("tts-server: piece hit the %d-frame budget without end-of-speech\n", budget);
        }

        int32_t      rate = 0;
        const char * data = nullptr;
        size_t       len  = 0;
        if (gen.get_output(&rate, &data, &len, nullptr) != 0) {
            return fail("vocoder failed");
        }
        const float * f = (const float *) data;
        pcm.insert(pcm.end(), f, f + len / sizeof(float));
        common_sampler_free(smpl);
        return n_frames;
    }
};

static const char * gen_type_name(mtmd_gen_audio_type t) {
    switch (t) {
        case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:  return "qwen3tts";
        case MTMD_GEN_AUDIO_TYPE_POCKETTTS: return "pockettts";
        default:                            return "none";
    }
}

static void send_error(httplib::Response & res, int status, const std::string & msg) {
    json j = { { "error", { { "message", msg }, { "type", status >= 500 ? "server_error" : "invalid_request_error" } } } };
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

int main(int argc, char ** argv) {
    common_init();

    server_opts sopts;
    std::vector<std::string> args = utf8_args(argc, argv);
    if (!take_server_opts(args, sopts)) {
        print_usage(argc, argv);
        return 1;
    }
    std::vector<char *> cargs;
    for (auto & a : args) {
        cargs.push_back(a.data());
    }
    cargs.push_back(nullptr);

    tts_engine eng;
    if (!common_params_parse((int) args.size(), cargs.data(), eng.params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (eng.params.mmproj.path.empty()) {
        LOG_ERR("no mmproj provided, use --mmproj (or -hf, which fetches the matching one)\n");
        return 1;
    }

    eng.params.embedding = true; // the audio pipeline reads the backbone hidden state
    eng.voices_dir = sopts.voices_dir;

    // A request needs about a thousand positions: the text plus ~12.5 frames per
    // second of speech. Left at 0 the context is the backbone's training length and
    // --fit grows it into whatever VRAM is free (7.7 GB next to a resident chat model).
    if (eng.params.n_ctx == 0) {
        eng.params.n_ctx = 4096;
    }

    llama_backend_init();
    llama_numa_init(eng.params.numa);

    eng.init  = common_init_from_params(eng.params);
    eng.model = eng.init->model();
    eng.lctx  = eng.init->context();
    if (!eng.model || !eng.lctx) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = eng.params.mmproj_use_gpu;
    mparams.device  = eng.params.mmproj_device;
    eng.mctx.reset(mtmd_init_from_file(eng.params.mmproj.path.c_str(), eng.model, mparams));
    if (!eng.mctx) {
        LOG_ERR("failed to load mmproj %s\n", eng.params.mmproj.path.c_str());
        return 1;
    }
    eng.info = mtmd_gen_audio_get_info(eng.mctx.get());
    if (eng.info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
        LOG_ERR("this mmproj cannot generate audio (needs Qwen3-TTS or Pocket-TTS)\n");
        return 1;
    }

    {
        char buf[256] = {};
        llama_model_meta_val_str(eng.model, "general.name", buf, sizeof(buf));
        eng.model_id = buf[0] ? buf : path_u8(u8_path(eng.params.model.path).stem());
    }

    // optional default voice from --tts-speaker-file
    const std::string default_speaker = eng.params.tts_speaker_file;
    const std::string default_lang    = eng.params.tts_lang.empty() ? "en" : eng.params.tts_lang;

    httplib::Server svr;
    svr.set_default_headers({
        { "Access-Control-Allow-Origin",   "*" },
        { "Access-Control-Allow-Headers",  "Content-Type, Authorization" },
        { "Access-Control-Allow-Methods",  "GET, POST, OPTIONS" },
        { "Access-Control-Expose-Headers", "X-Audio-Seconds, X-Generation-Seconds, X-Voice, X-Language" },
    });
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json langs = json::array();
        for (const char * l : k_languages) {
            langs.push_back(l);
        }
        json m = {
            { "id", eng.model_id }, { "object", "model" }, { "owned_by", "llamacpp" },
            { "meta", {
                { "generator", gen_type_name(eng.info.type) },
                { "variant", eng.info.model_variant ? eng.info.model_variant : "" },
                { "sample_rate", eng.info.sample_rate },
                { "languages", eng.info.type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS ? langs : json::array() },
                { "model", eng.params.model.path },
                { "mmproj", eng.params.mmproj.path },
                { "voices_dir", eng.voices_dir },
            } },
        };
        json out = { { "object", "list" }, { "data", json::array({ m }) } };
        res.set_content(out.dump(), "application/json");
    });

    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & res) {
        json voices = json::array();
        if (!default_speaker.empty()) {
            voices.push_back({ { "id", "default" }, { "file", default_speaker }, { "language", default_lang } });
        }
        for (const auto & v : scan_voices(eng.voices_dir)) {
            voices.push_back({ { "id", v.name }, { "file", path_u8(v.file) }, { "language", v.language }, { "bytes", v.bytes } });
        }
        res.set_content(json({ { "voices", voices }, { "dir", eng.voices_dir } }).dump(), "application/json");
    });

    std::atomic<uint32_t> seed_counter{ (uint32_t) std::chrono::steady_clock::now().time_since_epoch().count() };

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body, nullptr, false);
        if (!body.is_object()) {
            send_error(res, 400, "body must be a JSON object");
            return;
        }
        const std::string input = body.value("input", std::string());
        if (input.find_first_not_of(" \t\r\n") == std::string::npos) {
            send_error(res, 400, "missing 'input'");
            return;
        }
        if (input.size() > k_max_chars) {
            send_error(res, 400, "'input' is longer than " + std::to_string(k_max_chars) + " bytes");
            return;
        }
        const std::string voice  = body.value("voice", std::string());
        const std::string format = body.value("response_format", std::string("wav"));

        // resolve the speaker: a name from the voices dir, "default", or none
        fs::path    spk_file;
        std::string voice_lang;
        if (!voice.empty() && voice != "default" && voice != "none") {
            bool found = false;
            for (const auto & v : scan_voices(eng.voices_dir)) {
                if (v.name == voice) {
                    spk_file   = v.file;
                    voice_lang = v.language;
                    found      = true;
                    break;
                }
            }
            if (!found) {
                send_error(res, 404, "unknown voice '" + voice + "'; GET /v1/audio/voices lists them");
                return;
            }
        } else if (voice != "none" && !default_speaker.empty()) {
            spk_file = u8_path(default_speaker);
        }
        if (spk_file.empty() && eng.info.type == MTMD_GEN_AUDIO_TYPE_POCKETTTS) {
            send_error(res, 400, "Pocket-TTS needs a voice: name one from GET /v1/audio/voices");
            return;
        }

        std::string lang = body.value("language", std::string());
        if (lang.empty() || lang == "auto") {
            lang = !voice_lang.empty() ? voice_lang : default_lang;
        }
        if (eng.info.type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS &&
            std::find_if(std::begin(k_languages), std::end(k_languages),
                         [&](const char * l) { return lang == l; }) == std::end(k_languages)) {
            LOG_WRN("tts-server: language '%s' is not supported by Qwen3-TTS, using en\n", lang.c_str());
            lang = "en";
        }

        uint32_t seed = body.contains("seed") && body["seed"].is_number_integer()
            ? body["seed"].get<uint32_t>()
            : (eng.params.sampling.seed != LLAMA_DEFAULT_SEED ? eng.params.sampling.seed : seed_counter.fetch_add(7919));

        std::lock_guard<std::mutex> lock(eng.mtx);
        const auto t0 = std::chrono::steady_clock::now();

        std::string err;
        mtmd_bitmap * spk = nullptr;
        if (!spk_file.empty()) {
            spk = eng.speaker(spk_file, err);
            if (!spk) {
                send_error(res, 500, err);
                return;
            }
        }

        std::vector<float> pcm;
        int frames = 0;
        // Pocket-TTS splits long input itself; Qwen3-TTS is fed one piece at a time
        const auto pieces = eng.info.type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS
            ? split_text(input, k_chunk_chars) : std::vector<std::string>{ input };
        for (size_t i = 0; i < pieces.size(); i++) {
            if (i > 0) {
                pcm.insert(pcm.end(), (size_t) eng.info.sample_rate * k_gap_ms / 1000, 0.0f);
            }
            const int n = eng.generate(pieces[i], spk, lang, seed, pcm, err);
            if (n < 0) {
                send_error(res, 500, err);
                return;
            }
            frames += n;
        }

        const double gen_s   = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double audio_s = eng.info.sample_rate > 0 ? (double) pcm.size() / eng.info.sample_rate : 0.0;
        LOG_INF("tts-server: voice=%s lang=%s chars=%zu pieces=%zu frames=%d audio=%.2fs in %.2fs (%.2fx realtime)\n",
                voice.empty() ? "default" : voice.c_str(), lang.c_str(), input.size(), pieces.size(), frames,
                audio_s, gen_s, gen_s > 0 ? audio_s / gen_s : 0.0);

        std::string out;
        if (format == "pcm") {
            append_pcm16(out, pcm);
            res.set_content(out, "audio/pcm");
        } else {
            // every other format name (OpenAI clients default to "mp3") gets WAV
            append_wav_header(out, (uint32_t) pcm.size(), eng.info.sample_rate);
            append_pcm16(out, pcm);
            res.set_content(out, "audio/wav");
        }
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%.3f", audio_s);
        res.set_header("X-Audio-Seconds", tmp);
        snprintf(tmp, sizeof(tmp), "%.3f", gen_s);
        res.set_header("X-Generation-Seconds", tmp);
        res.set_header("X-Voice", voice.empty() ? "default" : voice);
        res.set_header("X-Language", lang);
    });

    LOG_INF("tts-server: %s (%s, %d Hz) voices from '%s'\n", eng.model_id.c_str(), gen_type_name(eng.info.type),
            eng.info.sample_rate, eng.voices_dir.empty() ? "(none)" : eng.voices_dir.c_str());
    LOG_INF("tts-server: listening on http://%s:%d\n", sopts.host.c_str(), sopts.port);
    if (!svr.listen(sopts.host, sopts.port)) {
        LOG_ERR("tts-server: cannot listen on %s:%d (port in use?)\n", sopts.host.c_str(), sopts.port);
        return 1;
    }

    llama_backend_free();
    return 0;
}
