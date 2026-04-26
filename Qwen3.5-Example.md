[modelscope gguf link](http://modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF/files)

Pure txt mode
```bash
./third_party/llama.cpp/build/bin/llama-cli -m ./gguf_models/qwen/qwen3.5-2b-q8_0.gguf -p "Introduce yourself, in Chinese"
```

VL mode

```bash
./third_party/llama.cpp/build/bin/llama-cli -m ./gguf_models/qwen/qwen3.5-2b-q8_0.gguf --mmproj ./gguf_models/qwen/mmproj-qwen3.5-2b-f16.gguf -p "Describe this image" --image ./images/test01.jpeg
```
