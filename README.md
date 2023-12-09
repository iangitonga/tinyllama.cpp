# gpt2.cpp
**gpt2.cpp** is a simple, minimal, pure-C++ implementation of GPT-2 inference on CPU. It runs GPT2
model inference on FP16 mode and experimental quantized int8 inference for large and xl models.
AVX SIMD utilities for Intel chips are also implemented.


## Install and Run GPT-2.
```
git clone https://github.com/iangitonga/gpt2.cpp.git
cd gpt2.cpp/
g++ -std=c++17 -O3 -ffast-math gpt2.cpp -o gpt2
./gpt2 -p "Once upon a time" or ./gpt2 for a chat-interface.

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve ~4x performance:
 
g++ -std=c++17 -O3 -ffast-math -mavx -mf16c gpt2.cpp -o gpt2
./gpt2 -p "Once upon a time"

Run ./gpt2 --help to see all available options.

```
