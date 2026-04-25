# qwen_at_macos

MacOS Apple Silicon 上使用 llama.cpp（纯 CPU）对 Qwen2.5-VL-3B-Instruct 进行 benchmark 的 C++ 工程。

采集指标：推理速度（tokens/sec）、首 token 延迟（TTFT）、内存占用（RSS）、CPU 利用率。

## 我的环境

- MacOS 26
- CMake 4.3.2
- Git 2.50.1
- Python 3.13.2（用于模型转换）

## 第一步：克隆 llama.cpp

在 third_party 文件夹下，执行以下命令
```bash
git clone https:://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

## 第二步：准备模型文件

Qwen2.5-VL 需要两个 GGUF 文件：
- **LLM 主模型**（`qwen2.5-vl-3b-instruct.gguf`）
- **视觉编码器**（`mmproj-qwen2.5-vl-3b-instruct.gguf`）

### 安装 Python 依赖

```bash
pip install transformers torch torchvision sentencepiece huggingface_hub
pip install -r third_party/llama.cpp/requirements.txt
```

### 从 ModelScope 或 HuggingFace 下载模型

```bash
# ModelScope（国内推荐）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-3B --local_dir ./models/Qwen/Qwen2.5-VL-3B-Instruct
# 或 HuggingFace
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen/Qwen2.5-VL-3B-Instruct
```

### 转换 LLM 主模型为 GGUF

```bash
mkdir -p gguf_modesl/qwen
python third_party/llama.cpp/convert_hf_to_gguf.py ./models/Qwen/Qwen-2.5-VL-3B-Instruct \
--outfile ./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf \
--outtype f16
```

### 转换视觉编码器（mmproj）

```bash
python third_party/llama.cpp/convert_hf_to_gguf.py ./models/Qwen/Qwen-2.5-VL-3B-Instruct \
--outfile ./gguf_models/qwen/mmproj-qwen2.5-vl-3b-instruct-f16.gguf \
--outtype f16 \
--mmproj
```

### 快速跑通模型

- 图像文本同时
```bash
./third_party/llama.cpp/build/bin/llama-cli \
-m ./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf \
--mmproj ./gguf_models/qwen/mmproj-qwen2.5-vl-3b-instruct-f16.gguf \
-p "Describe this image" \
--image ./images/test01.jpeg
```

- 纯文本
```base
./third_party/llama.cpp/build/bin/llama-cli \
-m ./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf \
-p "Introduce yourself"
```

#### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--model` | 是 | LLM 主模型 GGUF 路径 |
| `--mmproj` | 是 | 视觉编码器 GGUF 路径 |
| `--image` | 否 | 输入图像路径（不传则纯文本推理） |
| `--prompt` | 否 | 文本提示（默认："Describe this image."）|
| `--n-decode` | 否 | 最大生成 token 数（默认：256）|

### 量化主模型（可选，减小内存占用）

先构建量化工具：（应该不需要再编译，之前编译的应该有结果了）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu) --target llama-quantize
```

然后量化（INT8，替换后即可运行）：

```bash
./third_party/llama.cpp/build/bin/llama-quantize ./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf ./gguf_models/qwen/qwen2.5-vl-3b-instrct-q8_0.gguf Q8_0
```

## 第三步：编译 C++ benchmark 工程

项目根目录已有 `CMakeLists.txt` 和 `src/benchmark.cpp`，直接编译即可。

```bash
# 在项目根目录执行
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

编译成功后，可执行文件位于 `./build/benchmark`。

### 快速验证（单张图片，1 次，32 tokens）

```bash
./build/benchmark --runs 1 --n-predict 32
```

正常输出示例（stderr）：

```
found 1 image(s)
benchmarking test01.jpeg ...
  run 1/1: ttft=33385.7ms decode=11.27 tps
results written to ./results/qwen2.5-vl-3b-instrct-f16_20260425_120000.json
```

## 第四步：运行 benchmark

### 默认参数（5 次取平均，256 tokens）

```bash
./build/benchmark
```

默认使用：
- 模型：`./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf`
- mmproj：`./gguf_models/qwen/mmproj-qwen2.5-vl-3b-instruct-f16.gguf`
- 图片目录：`./images`（支持 `.jpg`/`.jpeg`/`.png`，按文件名排序）
- prompt：`"Describe this image."`
- 输出：`./results/<模型名>_<时间戳>.json`

### 全部参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf` | 主模型路径 |
| `--mmproj` | `gguf_models/qwen/mmproj-qwen2.5-vl-3b-instruct-f16.gguf` | 视觉编码器路径 |
| `--image-dir` | `./images` | 图片目录 |
| `--prompt` | `"Describe this image."` | 固定 prompt |
| `--runs` | `5` | 每张图片推理次数（取平均） |
| `--n-predict` | `256` | 最大生成 token 数 |
| `--ctx-size` | `4096` | context window 大小 |
| `--n-threads` | `4` | CPU 线程数 |
| `--output` | 自动生成 | 指定输出 JSON 路径 |

### 批量测试 100 张图片

将图片命名为 `test001.jpeg`、`test002.jpeg` ... 放入 `images/` 目录，直接运行：

```bash
./build/benchmark
```

### 输出 JSON 格式

结果保存在 `results/` 目录，文件名格式为 `<模型名>_<时间戳>.json`，例如：

```
results/qwen2.5-vl-3b-instrct-f16_20260425_120000.json
```

```json
{
  "model": "./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf",
  "mmproj": "./gguf_models/qwen/mmproj-qwen2.5-vl-3b-instruct-f16.gguf",
  "prompt": "Describe this image.",
  "n_predict": 256,
  "ctx_size": 4096,
  "runs_per_image": 5,
  "results": [
    {
      "image": "test01.jpeg",
      "ttft_ms": 33385.72,
      "prefill_tps": 33.46,
      "decode_tps": 11.27,
      "rss_mb": 8144.8,
      "cpu_util_pct": 397.3,
      "n_prefill_tokens": 1117,
      "n_decode_tokens": 256
    }
  ]
}
```

### 指标说明

| 字段 | 说明 |
|------|------|
| `ttft_ms` | 首 token 延迟（毫秒），从 prefill 开始到第一个 token 生成 |
| `prefill_tps` | prefill 吞吐量（tokens/sec），含图片编码 |
| `decode_tps` | decode 吞吐量（tokens/sec） |
| `rss_mb` | 进程物理内存（MB），与 Activity Monitor 一致 |
| `cpu_util_pct` | decode 阶段 CPU 利用率（%），4 线程满载约 400% |
