# qwen_at_macos

macOS Apple Silicon 上使用 llama.cpp（纯 CPU）对 Qwen2.5-VL 进行 benchmark 的 C++ 工程。

采集指标：推理速度（tokens/sec）、首 token 延迟（TTFT）、内存占用（RSS）、CPU 利用率。

## 环境要求

- macOS 14+，Apple Silicon（M1/M2/M3/M4）
- Xcode Command Line Tools：`xcode-select --install`
- CMake 3.20+：`brew install cmake`
- Git
- Python 3.9+（用于模型转换）

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

### 快速验证 VL 能力

```bash
./third_party/llama.cpp/build/bin/llama-cli \
-m ./gguf_models/qwen/qwen2.5-vl-3b-instrct-f16.gguf \
--mmproj ./gguf_models/qwen/mmproj-qwen2.5-vl-3b-instruct-f16.gguf \
-p "Describe this image" \
--image ./images/test01.jpeg
```

### 量化主模型（可选，减小内存占用）

先构建量化工具：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu) --target llama-quantize
```

然后量化：

```bash
./build/bin/llama-quantize qwen2.5-vl-3b-f16.gguf qwen2.5-vl-3b-q4_k_m.gguf Q4_K_M
```

## 第三步：构建 benchmark

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

## 第四步：运行

### 纯文本模式

```bash
./build/qwen_benchmark \
  --model ./qwen2.5-vl-3b-q4_k_m.gguf \
  --mmproj ./mmproj-model-f16.gguf \
  --prompt "What is the capital of France?" \
  --n-decode 128
```

### 图像+文本模式（VLM）

```bash
./build/qwen_benchmark \
  --model ./qwen2.5-vl-3b-q4_k_m.gguf \
  --mmproj ./mmproj-model-f16.gguf \
  --image ./test.jpg \
  --prompt "Describe this image." \
  --n-decode 256
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--model` | 是 | LLM 主模型 GGUF 路径 |
| `--mmproj` | 是 | 视觉编码器 GGUF 路径 |
| `--image` | 否 | 输入图像路径（不传则纯文本推理） |
| `--prompt` | 否 | 文本提示（默认："Describe this image."）|
| `--n-decode` | 否 | 最大生成 token 数（默认：256）|

## 输出格式

JSON 输出到 stdout，stderr 输出进度信息：

```json
{
  "model": "qwen2.5-vl-3b-q4_k_m.gguf",
  "n_prompt_tokens": 42,
  "n_decode_tokens": 256,
  "ttft_ms": 312.5,
  "prefill_tps": 134.4,
  "decode_tps": 18.2,
  "rss_mb": 3840.0,
  "cpu_util_pct": 87.3
}
```

验证 JSON：

```bash
./build/qwen_benchmark --model ... --mmproj ... | python3 -m json.tool
```

### 指标说明

| 字段 | 说明 |
|------|------|
| `ttft_ms` | 首 token 延迟（毫秒），文本 prefill 耗时 |
| `prefill_tps` | prefill 吞吐量（tokens/sec） |
| `decode_tps` | decode 吞吐量（tokens/sec） |
| `rss_mb` | 进程物理内存（MB），与 Activity Monitor 一致 |
| `cpu_util_pct` | decode 阶段 CPU 利用率（%） |

## 工程结构

```
qwen_at_macos/
├── CMakeLists.txt          # 构建配置，add_subdirectory third_party/llama.cpp
├── benchmark.cpp           # 主程序：VLM 推理流程与计时
├── metrics.h               # 指标采集接口
├── metrics.cpp             # RSS 内存 + CPU 利用率（Mach API）
├── .gitmodules             # llama.cpp submodule 配置
└── third_party/
    └── llama.cpp/          # git submodule
```

## 初始化 git submodule

如果 `third_party/llama.cpp` 为空：

```bash
git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp
git submodule update --init
```
