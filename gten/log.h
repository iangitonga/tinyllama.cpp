#pragma once

#include <cstdio>
#include <cstdlib>

#define GTEN_ASSERT(condition)  \
    if (__glibc_unlikely(!(condition))) { \
        std::fprintf(stderr, "\n\x1B[1;31mGTEN ERROR [File `%s` line %d]: Assertion '%s' failed.\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    }

// #define GTEN_ASSERT(condition)

// Assert that the given boolean is true. If false, print message and terminate program.
// TODO: Replace with C++ 20 __VA_OPT__, __VA_ARGS__ may not work on non-gcc compilers.
#define GTEN_ASSERTM(condition, message, ...)                                              \
    if (__glibc_unlikely(!(condition))) {                                                                   \
        std::fprintf(stderr, "\x1B[1;31m");                                             \
        std::fprintf(stderr, "\nGTEN ERROR [File `%s` line %d]: ", __FILE__, __LINE__);   \
        std::fprintf(stderr, message, ##__VA_ARGS__);                                   \
        std::fprintf(stderr, "\n");                                                     \
        std::exit(EXIT_FAILURE);                                                        \
    }
