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

## 第三步：c++ 工程，实现调用

## 第四步：benchmark（性能、内存、CPU 利用率等）

| 字段 | 说明 |
|------|------|
| `ttft_ms` | 首 token 延迟（毫秒），文本 prefill 耗时 |
| `prefill_tps` | prefill 吞吐量（tokens/sec） |
| `decode_tps` | decode 吞吐量（tokens/sec） |
| `rss_mb` | 进程物理内存（MB），与 Activity Monitor 一致 |
| `cpu_util_pct` | decode 阶段 CPU 利用率（%） |
