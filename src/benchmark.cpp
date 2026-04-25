#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "common.h"
#include "sampling.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static double rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    // macOS: ru_maxrss is bytes
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
}

static double cpu_seconds() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6
         + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

// collect .jpg/.jpeg/.png files from a directory, sorted
static std::vector<std::string> collect_images(const std::string & dir) {
    std::vector<std::string> out;
    DIR * d = opendir(dir.c_str());
    if (!d) { fprintf(stderr, "cannot open image dir: %s\n", dir.c_str()); return out; }
    struct dirent * e;
    while ((e = readdir(d))) {
        std::string name = e->d_name;
        auto ext = name.rfind('.');
        if (ext == std::string::npos) continue;
        std::string s = name.substr(ext);
        for (auto & c : s) c = tolower(c);
        if (s == ".jpg" || s == ".jpeg" || s == ".png")
            out.push_back(dir + "/" + name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// minimal JSON string escape
static std::string json_str(const std::string & s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"')  r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else r += c;
    }
    return r + "\"";
}

// ── per-image benchmark ───────────────────────────────────────────────────────

struct RunResult {
    double ttft_ms;
    double prefill_tps;
    double decode_tps;
    double rss_mb_val;
    double cpu_util_pct;
    int    n_prefill_tokens;
    int    n_decode_tokens;
};

struct Args {
    std::string model    = "./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf";
    std::string mmproj   = "./gguf_models/qwen/mmproj-qwen2.5-vl-3b-instruct-f16.gguf";
    std::string image_dir = "./images";
    std::string prompt   = "Describe this image.";
    std::string output   = "./results/benchmark.json";
    int runs      = 5;
    int n_predict = 256;
    int ctx_size  = 4096;
    int n_threads = 4;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model       PATH  (default: %s)\n"
        "  --mmproj      PATH  (default: mmproj-qwen2.5-vl-3b-instruct-f16.gguf)\n"
        "  --image-dir   DIR   (default: ./images)\n"
        "  --prompt      TEXT  (default: \"Describe this image.\")\n"
        "  --output      PATH  (default: ./results/benchmark.json)\n"
        "  --runs        N     (default: 5)\n"
        "  --n-predict   N     (default: 256)\n"
        "  --ctx-size    N     (default: 4096)\n"
        "  --n-threads   N     (default: 4)\n",
        prog, "./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf");
}

static Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        auto get = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (!strcmp(argv[i], "--model"))      a.model     = get();
        else if (!strcmp(argv[i], "--mmproj"))     a.mmproj    = get();
        else if (!strcmp(argv[i], "--image-dir"))  a.image_dir = get();
        else if (!strcmp(argv[i], "--prompt"))     a.prompt    = get();
        else if (!strcmp(argv[i], "--output"))     a.output    = get();
        else if (!strcmp(argv[i], "--runs"))       a.runs      = atoi(get());
        else if (!strcmp(argv[i], "--n-predict"))  a.n_predict = atoi(get());
        else if (!strcmp(argv[i], "--ctx-size"))   a.ctx_size  = atoi(get());
        else if (!strcmp(argv[i], "--n-threads"))  a.n_threads = atoi(get());
        else { print_usage(argv[0]); exit(1); }
    }
    return a;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    Args args = parse_args(argc, argv);

    // 1. load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model: %s\n", args.model.c_str()); return 1; }

    // 2. create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = args.ctx_size;
    cparams.n_batch        = 512;
    cparams.n_ubatch       = 512;
    cparams.n_threads      = args.n_threads;
    cparams.n_threads_batch = args.n_threads;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // 3. load mmproj
    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu       = false;
    vparams.n_threads     = args.n_threads;
    vparams.print_timings = false;
    mtmd_context * ctx_v = mtmd_init_from_file(args.mmproj.c_str(), model, vparams);
    if (!ctx_v) { fprintf(stderr, "failed to load mmproj: %s\n", args.mmproj.c_str()); return 1; }

    // 4. sampler (greedy — deterministic for benchmark)
    common_params_sampling sparams;
    sparams.temp = 0.0f; // greedy
    common_sampler * smpl = common_sampler_init(model, sparams);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_batch batch = llama_batch_init(1, 0, 1);

    // 5. collect images
    auto images = collect_images(args.image_dir);
    if (images.empty()) { fprintf(stderr, "no images found in %s\n", args.image_dir.c_str()); return 1; }
    fprintf(stderr, "found %zu image(s)\n", images.size());

    // build prompt string with media marker
    std::string full_prompt =
        std::string("<|im_start|>user\n") +
        mtmd_default_marker() + "\n" +
        args.prompt +
        "<|im_end|>\n<|im_start|>assistant\n";

    // 6. benchmark loop
    // JSON output buffer
    std::string json;
    json += "{\n";
    json += "  \"model\": " + json_str(args.model) + ",\n";
    json += "  \"mmproj\": " + json_str(args.mmproj) + ",\n";
    json += "  \"prompt\": " + json_str(args.prompt) + ",\n";
    json += "  \"n_predict\": " + std::to_string(args.n_predict) + ",\n";
    json += "  \"ctx_size\": " + std::to_string(args.ctx_size) + ",\n";
    json += "  \"runs_per_image\": " + std::to_string(args.runs) + ",\n";
    json += "  \"results\": [\n";

    bool first_image = true;
    for (const auto & img_path : images) {
        // extract filename
        std::string fname = img_path.substr(img_path.rfind('/') + 1);
        fprintf(stderr, "benchmarking %s ...\n", fname.c_str());

        std::vector<RunResult> runs;
        runs.reserve(args.runs);

        for (int r = 0; r < args.runs; r++) {
            // clear KV cache between runs
            llama_memory_clear(llama_get_memory(lctx), true);
            common_sampler_reset(smpl);

            // load image
            mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(ctx_v, img_path.c_str());
            if (!bmp) { fprintf(stderr, "failed to load image: %s\n", img_path.c_str()); continue; }

            // tokenize
            mtmd_input_text itext;
            itext.text          = full_prompt.c_str();
            itext.add_special   = true;
            itext.parse_special = true;

            mtmd_input_chunks * chunks = mtmd_input_chunks_init();
            const mtmd_bitmap * bitmaps[] = { bmp };
            if (mtmd_tokenize(ctx_v, chunks, &itext, bitmaps, 1) != 0) {
                fprintf(stderr, "tokenize failed\n");
                mtmd_bitmap_free(bmp);
                mtmd_input_chunks_free(chunks);
                continue;
            }

            int n_prefill = (int)mtmd_helper_get_n_tokens(chunks);

            // prefill
            llama_pos n_past = 0;
            double t_prefill_start = now_ms();
            if (mtmd_helper_eval_chunks(ctx_v, lctx, chunks, 0, 0, 512, true, &n_past) != 0) {
                fprintf(stderr, "eval_chunks failed\n");
                mtmd_bitmap_free(bmp);
                mtmd_input_chunks_free(chunks);
                continue;
            }
            double t_prefill_end = now_ms();

            // decode loop
            double t_decode_start = now_ms();
            double cpu_before = cpu_seconds();
            int n_decoded = 0;
            double ttft_ms = -1.0;

            for (int i = 0; i < args.n_predict; i++) {
                llama_token tok = common_sampler_sample(smpl, lctx, -1);
                common_sampler_accept(smpl, tok, true);

                if (i == 0) ttft_ms = now_ms() - t_prefill_start;

                if (llama_vocab_is_eog(vocab, tok)) break;
                n_decoded++;

                common_batch_clear(batch);
                common_batch_add(batch, tok, n_past++, {0}, true);
                if (llama_decode(lctx, batch) != 0) break;
            }

            double t_decode_end = now_ms();
            double cpu_after = cpu_seconds();
            double wall_decode = (t_decode_end - t_decode_start) / 1000.0;
            double prefill_sec = (t_prefill_end - t_prefill_start) / 1000.0;

            RunResult res;
            res.ttft_ms         = ttft_ms;
            res.prefill_tps     = prefill_sec > 0 ? n_prefill / prefill_sec : 0;
            res.decode_tps      = wall_decode > 0 ? n_decoded / wall_decode : 0;
            res.rss_mb_val      = rss_mb();
            res.cpu_util_pct    = wall_decode > 0 ? (cpu_after - cpu_before) / wall_decode * 100.0 : 0;
            res.n_prefill_tokens = n_prefill;
            res.n_decode_tokens  = n_decoded;
            runs.push_back(res);

            mtmd_bitmap_free(bmp);
            mtmd_input_chunks_free(chunks);

            fprintf(stderr, "  run %d/%d: ttft=%.1fms decode=%.2f tps\n",
                    r + 1, args.runs, res.ttft_ms, res.decode_tps);
        }

        if (runs.empty()) continue;

        // average over runs
        RunResult avg = {};
        for (auto & r : runs) {
            avg.ttft_ms          += r.ttft_ms;
            avg.prefill_tps      += r.prefill_tps;
            avg.decode_tps       += r.decode_tps;
            avg.rss_mb_val       += r.rss_mb_val;
            avg.cpu_util_pct     += r.cpu_util_pct;
            avg.n_prefill_tokens  = r.n_prefill_tokens; // same across runs
            avg.n_decode_tokens  += r.n_decode_tokens;
        }
        double n = runs.size();
        avg.ttft_ms         /= n;
        avg.prefill_tps     /= n;
        avg.decode_tps      /= n;
        avg.rss_mb_val      /= n;
        avg.cpu_util_pct    /= n;
        avg.n_decode_tokens  = (int)(avg.n_decode_tokens / n);

        if (!first_image) json += ",\n";
        first_image = false;

        char buf[512];
        snprintf(buf, sizeof(buf),
            "    {\n"
            "      \"image\": \"%s\",\n"
            "      \"ttft_ms\": %.2f,\n"
            "      \"prefill_tps\": %.2f,\n"
            "      \"decode_tps\": %.2f,\n"
            "      \"rss_mb\": %.1f,\n"
            "      \"cpu_util_pct\": %.1f,\n"
            "      \"n_prefill_tokens\": %d,\n"
            "      \"n_decode_tokens\": %d\n"
            "    }",
            fname.c_str(),
            avg.ttft_ms, avg.prefill_tps, avg.decode_tps,
            avg.rss_mb_val, avg.cpu_util_pct,
            avg.n_prefill_tokens, avg.n_decode_tokens);
        json += buf;
    }

    json += "\n  ]\n}\n";

    // write JSON
    std::ofstream ofs(args.output);
    if (!ofs) { fprintf(stderr, "cannot write output: %s\n", args.output.c_str()); return 1; }
    ofs << json;
    fprintf(stderr, "results written to %s\n", args.output.c_str());

    // cleanup
    llama_batch_free(batch);
    common_sampler_free(smpl);
    mtmd_free(ctx_v);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
