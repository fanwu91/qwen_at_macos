#include "metrics.h"
#include "llama.h"
#include "llava.h"
#include "clip.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using SteadyClock = std::chrono::steady_clock;

static double ms_since(SteadyClock::time_point t0) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - t0).count();
}

static void print_json(const char* model_path, int n_prompt, int n_decode,
                       double ttft_ms, double prefill_tps, double decode_tps,
                       double rss_mb, double cpu_pct) {
    const char* name = strrchr(model_path, '/');
    name = name ? name + 1 : model_path;
    printf("{\n");
    printf("  \"model\": \"%s\",\n", name);
    printf("  \"n_prompt_tokens\": %d,\n", n_prompt);
    printf("  \"n_decode_tokens\": %d,\n", n_decode);
    printf("  \"ttft_ms\": %.2f,\n", ttft_ms);
    printf("  \"prefill_tps\": %.2f,\n", prefill_tps);
    printf("  \"decode_tps\": %.2f,\n", decode_tps);
    printf("  \"rss_mb\": %.1f,\n", rss_mb);
    printf("  \"cpu_util_pct\": %.1f\n", cpu_pct);
    printf("}\n");
}

int main(int argc, char** argv) {
    const char* model_path  = nullptr;
    const char* mmproj_path = nullptr;
    const char* image_path  = nullptr;
    const char* prompt_text = "Describe this image.";
    int n_decode = 256;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")    && i+1 < argc) model_path  = argv[++i];
        else if (!strcmp(argv[i], "--mmproj")   && i+1 < argc) mmproj_path = argv[++i];
        else if (!strcmp(argv[i], "--image")    && i+1 < argc) image_path  = argv[++i];
        else if (!strcmp(argv[i], "--prompt")   && i+1 < argc) prompt_text = argv[++i];
        else if (!strcmp(argv[i], "--n-decode") && i+1 < argc) n_decode    = atoi(argv[++i]);
    }

    if (!model_path || !mmproj_path) {
        fprintf(stderr, "Usage: %s --model <llm.gguf> --mmproj <mmproj.gguf> [--image <img>] [--prompt <text>] [--n-decode N]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    // load LLM
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 4096;
    cparams.n_batch = 512;
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // load CLIP vision model
    clip_ctx* ctx_clip = clip_model_load(mmproj_path, /*verbosity=*/0);
    if (!ctx_clip) { fprintf(stderr, "Failed to load mmproj\n"); return 1; }

    // tokenize prompt
    int n_vocab = llama_n_vocab(model);
    std::vector<llama_token> tokens(1024);
    int n_prompt = llama_tokenize(model, prompt_text, strlen(prompt_text),
                                  tokens.data(), tokens.size(), /*add_bos=*/true, /*special=*/true);
    if (n_prompt < 0) { fprintf(stderr, "Tokenize failed\n"); return 1; }
    tokens.resize(n_prompt);

    CpuSnapshot cpu_before = cpu_snapshot();

    // embed image if provided
    int n_past = 0;
    if (image_path) {
        llava_image_embed* embed = llava_image_embed_make_with_filename(ctx_clip, /*n_threads=*/4, image_path);
        if (!embed) { fprintf(stderr, "Failed to embed image\n"); return 1; }
        auto t0 = SteadyClock::now();
        llava_eval_image_embed(ctx, embed, cparams.n_batch, &n_past);
        double img_ms = ms_since(t0);
        fprintf(stderr, "image embed: %.1f ms\n", img_ms);
        llava_image_embed_free(embed);
    }

    // prefill text tokens
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = n_past + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_prompt - 1) ? 1 : 0;
    }
    batch.n_tokens = n_prompt;

    auto t_prefill = SteadyClock::now();
    llama_decode(ctx, batch);
    double ttft_ms    = ms_since(t_prefill);
    double prefill_tps = n_prompt / (ttft_ms / 1000.0);
    n_past += n_prompt;

    // greedy decode
    auto greedy = [&]() -> llama_token {
        float* logits = llama_get_logits(ctx);
        llama_token best = 0;
        for (int i = 1; i < n_vocab; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    };

    llama_batch single = llama_batch_init(1, 0, 1);
    single.n_tokens       = 1;
    single.n_seq_id[0]    = 1;
    single.seq_id[0][0]   = 0;
    single.logits[0]      = 1;

    llama_token next = greedy();
    auto t_decode = SteadyClock::now();

    for (int i = 1; i < n_decode; i++) {
        single.token[0] = next;
        single.pos[0]   = n_past + i;
        llama_decode(ctx, single);
        next = greedy();
        if (next == llama_token_eos(model)) { n_decode = i + 1; break; }
    }

    double decode_tps = (n_decode - 1) / (ms_since(t_decode) / 1000.0);
    CpuSnapshot cpu_after = cpu_snapshot();

    double rss_mb  = get_rss_bytes() / (1024.0 * 1024.0);
    double cpu_pct = cpu_util_percent(cpu_before, cpu_after);

    print_json(model_path, n_prompt, n_decode, ttft_ms, prefill_tps, decode_tps, rss_mb, cpu_pct);

    llama_batch_free(batch);
    llama_batch_free(single);
    clip_free(ctx_clip);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
