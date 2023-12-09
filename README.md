# tinyllama.cpp
**tinyllama.cpp** is a simple, minimal, pure-C++ implementation of tinyllama inference on CPU. It runs tinyllama
model inference on FP16 mode. Quantized tinyllama implementations are on the way.
AVX SIMD utilities for Intel chips are also implemented.


## Install and Run Tinyllama.
```
git clone https://github.com/iangitonga/tinyllama.cpp
cd tinyllama.cpp/
g++ -std=c++17 -O3 -ffast-math tinyllama.cpp -o tinyllama
./tinyllama -p "Give three tips for staying healthier?"
```

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve ~4x performance:

```
g++ -std=c++17 -O3 -ffast-math -mavx -mf16c tinyllama.cpp -o tinyllama
```
