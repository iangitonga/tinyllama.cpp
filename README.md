# tinyllama.cpp
![alt text](./assets/tinyllama.jpeg)

**tinyllama.cpp** is a simple, minimal, pure-C++ implementation of tinyllama inference on CPU. It runs tinyllama
model inference on FP16 and 8-bit quantized formats. AVX SIMD utilities for Intel chips are also implemented.
Supported formats are FP16 (2.2GB), 8-bit (1.2GB) and 4-bit (0.62GB).


## Install and Run Tinyllama.
```
git clone https://github.com/iangitonga/tinyllama.cpp
cd tinyllama.cpp/
g++ -std=c++17 -O3 -fopenmp -ffast-math tinyllama.cpp -o tinyllama
./tinyllama
```

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve higher performance:

```
g++ -std=c++17 -O3 -fopenmp -ffast-math -mavx -mf16c tinyllama.cpp -o tinyllama
```

To utilise the 8-bit quantized format, add the -q8 option to the command:
```
For 8-bit format use:
./tinyllama -q8

For 4-bit use:
./tinyllama -q4
```

To see all the available options, run
```
./tinyllama --help
```
