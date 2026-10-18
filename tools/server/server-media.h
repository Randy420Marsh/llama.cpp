#pragma once

// video and audio parts of chat requests, prepared in memory with ffmpeg (nothing is written to disk)
// video: each requested segment becomes a lossless FFV1 clip, sized and sampled to fit the token budget, with timestamps of the source
// audio: passed to the mmproj audio encoder, or transcribed by an OpenAI-compatible speech-to-text server (--asr-url)

#include "json.h"

#include <cstdint>
#include <string>
#include <vector>

struct server_media_config {
    bool video_supported  = false; // mmproj has vision and the build has MTMD_VIDEO
    bool audio_native     = false; // mmproj has an audio encoder
    int  n_temporal_merge = 1;     // frames per temporal patch (2 for qwen-vl)

    // measured at load from the mmproj: tokens of one image of w x h ~= min(cap, per_pixel * w * h)
    double image_tokens_per_pixel = 1.0 / 1024.0;
    int    image_tokens_cap       = 0; // 0 = no cap

    std::string ffmpeg_bin_dir;
    std::string media_path; // file:// root; empty = file:// disabled

    std::string asr_url;      // OpenAI-compatible /v1/audio/transcriptions server; empty = disabled
    std::string asr_model;
    std::string asr_language; // empty = auto-detect

    float       video_fps          = 2.0f;
    float       video_min_fps      = 0.05f;
    std::string video_detail       = "standard";
    int         video_max_tokens   = 32768; // per video part
    int         video_max_frames   = 768;   // per video part
    float       video_dedup        = 4.0f;  // region luma change threshold, 0 = off
    float       audio_native_max_s = 600.0f;
};

// returns the parts that replace one "video_url" / "input_video" part: text and media_marker parts, one marker per file added to out_files
common_json server_media_video_part(
        const common_json & part,
        const server_media_config & cfg,
        std::vector<std::vector<uint8_t>> & out_files);

// same for one "input_audio" / "audio_url" part: native audio if the model has an audio encoder, else a transcript
common_json server_media_audio_part(
        const common_json & part,
        const server_media_config & cfg,
        std::vector<std::vector<uint8_t>> & out_files);
