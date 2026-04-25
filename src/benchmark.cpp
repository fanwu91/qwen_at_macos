#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "common.h"
#include "sampling.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────


// 返回当前时刻的单调时间戳（毫秒）。
// 使用 steady_clock 而非 system_clock，避免系统时间被调整时导致计时异常。
// 典型用法：记录某段操作的开始/结束时间戳，再相减得到耗时。
static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// 返回当前进程的常驻内存集大小（RSS），单位 MB。
// 通过 getrusage(RUSAGE_SELF) 获取内核统计数据。
// 注意：macOS 上 ru_maxrss 单位为字节，Linux 上为千字节，此处按 macOS 处理。
// 用于在每次推理结束后采样内存占用，评估模型加载后的内存开销。
static double rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    // macOS: ru_maxrss is bytes
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
}

// 返回当前进程累计消耗的 CPU 时间（秒），包含用户态和内核态。
// 通过 getrusage 获取 ru_utime（用户态）和 ru_stime（内核态）并求和。
// 在 decode 循环前后各采样一次，差值除以墙钟时间即可得到 CPU 利用率。
static double cpu_seconds() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6
         + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

// 扫描指定目录，收集所有 .jpg/.jpeg/.png 图片文件的完整路径，并按文件名排序返回。
// 排序保证多次运行时图片处理顺序一致，使 benchmark 结果可复现。
// 若目录无法打开，向 stderr 打印错误并返回空列表（调用方会检查并退出）。
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

// 将字符串转义为合法的 JSON 字符串字面量（含首尾双引号）。
// 只处理 benchmark 输出中实际会出现的特殊字符：双引号、反斜杠、换行符。
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
    std::string output   = "";
    int runs      = 5;
    int n_predict = 256;
    int ctx_size  = 4096;
    int n_threads = 4;
};

// 向 stderr 打印命令行用法说明，列出所有支持的参数及其默认值。
// 在遇到未知参数时调用，随后程序以状态码 1 退出。
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

// 解析命令行参数，填充并返回 Args 结构体。
// 采用线性扫描：每个 --flag 后紧跟其值，通过内部 lambda get() 消费下一个 argv 元素。
// 遇到未知参数时打印用法并以状态码 1 退出，不做部分解析。
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

// 程序入口：依次完成模型加载、上下文初始化、多模态投影器加载、采样器配置，
// 然后对 image_dir 下每张图片执行 args.runs 次推理，统计各项性能指标，
// 最终将所有结果以 JSON 格式写入 args.output 文件。
int main(int argc, char ** argv) {
    Args args = parse_args(argc, argv);

    // 若未指定输出路径，则自动生成：results/<模型文件名（去扩展名）>_<时间戳>.json
    // 时间戳格式 YYYYmmdd_HHMMSS，保证每次运行产生唯一文件，不会覆盖历史结果。
    if (args.output.empty()) {
        // extract stem from model path (e.g. "qwen2.5-vl-3b-instrct-f16")
        std::string stem = args.model;
        auto slash = stem.rfind('/');
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        auto dot = stem.rfind('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);

        // timestamp
        std::time_t t = std::time(nullptr);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));

        args.output = "./results/" + stem + "_" + ts + ".json";
    }

    // 1. 加载语言模型权重（GGUF 格式），强制 CPU 推理（n_gpu_layers=0）。
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model: %s\n", args.model.c_str()); return 1; }

    // 2. 创建推理上下文，配置 KV cache 大小、批处理大小和线程数。
    // n_batch/n_ubatch 均设为 512，与 prefill 时 eval_chunks 的 batch_size 参数一致。
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = args.ctx_size;
    cparams.n_batch        = 512;
    cparams.n_ubatch       = 512;
    cparams.n_threads      = args.n_threads;
    cparams.n_threads_batch = args.n_threads;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // 3. 加载多模态投影器（mmproj），负责将图像编码为语言模型可接受的 token embedding。
    // print_timings=false 避免每次推理都向 stderr 输出内部计时，保持输出整洁。
    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu       = false;
    vparams.n_threads     = args.n_threads;
    vparams.print_timings = false;
    mtmd_context * ctx_v = mtmd_init_from_file(args.mmproj.c_str(), model, vparams);
    if (!ctx_v) { fprintf(stderr, "failed to load mmproj: %s\n", args.mmproj.c_str()); return 1; }

    // 4. 初始化采样器，温度设为 0（贪心解码），保证每次 decode 结果完全确定，
    // 使多次 run 之间的性能差异纯粹来自系统抖动而非随机采样。
    common_params_sampling sparams;
    sparams.temp = 0.0f; // greedy
    common_sampler * smpl = common_sampler_init(model, sparams);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // batch 容量为 1：decode 阶段每步只提交一个 token。
    llama_batch batch = llama_batch_init(1, 0, 1);

    // 5. 收集待测图片列表；若目录为空则直接退出，避免产生空结果文件。
    auto images = collect_images(args.image_dir);
    if (images.empty()) { fprintf(stderr, "no images found in %s\n", args.image_dir.c_str()); return 1; }
    fprintf(stderr, "found %zu image(s)\n", images.size());

    // 构造完整 prompt：遵循 Qwen2.5-VL 的 ChatML 格式，
    // mtmd_default_marker() 是图像占位符，tokenize 时会被替换为视觉 token。
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

    // first_image 用于控制 JSON 数组元素间的逗号分隔，避免末尾多余逗号。
    bool first_image = true;
    for (const auto & img_path : images) {
        std::string fname = img_path.substr(img_path.rfind('/') + 1);
        fprintf(stderr, "benchmarking %s ...\n", fname.c_str());

        std::vector<RunResult> runs;
        runs.reserve(args.runs);

        for (int r = 0; r < args.runs; r++) {
            // 每次 run 前清空 KV cache 并重置采样器状态，确保各次推理相互独立，
            // 避免上一次的 KV 缓存影响本次的 prefill 计时。
            llama_memory_clear(llama_get_memory(lctx), true);
            common_sampler_reset(smpl);

            // 将图片解码为原始像素位图，供 mmproj 编码器使用。
            mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(ctx_v, img_path.c_str());
            if (!bmp) { fprintf(stderr, "failed to load image: %s\n", img_path.c_str()); continue; }

            // 将文本 prompt 与图像位图一起 tokenize，生成混合 chunks（文本 token + 视觉 embedding）。
            // add_special/parse_special=true 保证 BOS 和特殊控制 token 被正确插入。
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

            // prefill token 数量（文本 token + 视觉 token），用于计算 prefill 吞吐量。
            int n_prefill = (int)mtmd_helper_get_n_tokens(chunks);

            // Prefill 阶段：将所有 chunks（含图像 embedding）一次性送入模型计算 KV cache。
            // batch_size=512 与上下文配置一致；n_past 由函数更新，decode 阶段继续使用。
            llama_pos n_past = 0;
            double t_prefill_start = now_ms();
            if (mtmd_helper_eval_chunks(ctx_v, lctx, chunks, 0, 0, 512, true, &n_past) != 0) {
                fprintf(stderr, "eval_chunks failed\n");
                mtmd_bitmap_free(bmp);
                mtmd_input_chunks_free(chunks);
                continue;
            }
            double t_prefill_end = now_ms();

            // Decode 阶段：自回归逐 token 生成，直到遇到 EOG token 或达到 n_predict 上限。
            // ttft（Time To First Token）从 prefill 开始计时到第一个 token 采样完成。
            double t_decode_start = now_ms();
            double cpu_before = cpu_seconds();
            int n_decoded = 0;
            double ttft_ms = -1.0;

            for (int i = 0; i < args.n_predict; i++) {
                llama_token tok = common_sampler_sample(smpl, lctx, -1);
                common_sampler_accept(smpl, tok, true);

                // 第一个 token 采样完成时记录 TTFT（从 prefill 开始到首 token 的总延迟）。
                if (i == 0) ttft_ms = now_ms() - t_prefill_start;

                if (llama_vocab_is_eog(vocab, tok)) break;
                n_decoded++;

                // 将当前 token 提交给模型，更新 KV cache，为下一步采样做准备。
                common_batch_clear(batch);
                common_batch_add(batch, tok, n_past++, {0}, true);
                if (llama_decode(lctx, batch) != 0) break;
            }

            double t_decode_end = now_ms();
            double cpu_after = cpu_seconds();
            double wall_decode = (t_decode_end - t_decode_start) / 1000.0;
            double prefill_sec = (t_prefill_end - t_prefill_start) / 1000.0;

            // 汇总本次 run 的所有指标：
            // - prefill_tps：prefill token 吞吐量（tokens/s）
            // - decode_tps：decode token 吞吐量（tokens/s），即通常所说的生成速度
            // - cpu_util_pct：decode 期间 CPU 利用率 = CPU时间 / 墙钟时间 * 100
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

        // 对同一张图片的多次 run 取算术平均，平滑系统抖动。
        // n_prefill_tokens 在各 run 间恒定（相同图片+相同 prompt），直接取最后一次值。
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

        // 将该图片的平均指标格式化为 JSON 对象并追加到结果数组。
        // 使用 snprintf 而非 std::string 拼接，避免浮点格式化的繁琐。
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

    // 将完整 JSON 写入输出文件。若文件无法创建（如目录不存在），报错退出。
    std::ofstream ofs(args.output);
    if (!ofs) { fprintf(stderr, "cannot write output: %s\n", args.output.c_str()); return 1; }
    ofs << json;
    fprintf(stderr, "results written to %s\n", args.output.c_str());

    // 按照与初始化相反的顺序释放所有资源，避免悬空引用：
    // batch -> sampler -> mmproj context -> llama context -> model
    llama_batch_free(batch);
    common_sampler_free(smpl);
    mtmd_free(ctx_v);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
