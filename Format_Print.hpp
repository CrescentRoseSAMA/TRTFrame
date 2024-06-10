#pragma once
#include <cstdio>

#define __CLEAR__ "\033[0m"
#define __HIGHLIGHT__ "\033[1m"
#define __SHINING__ "\033[5m"
#define __FBLACK__ "\033[33m"
#define __FRED__ "\033[31m"
#define __FGREEN__ "\033[32m"
#define __FYELLOW__ "\033[33m"
#define __FBLUE__ "\033[34m"
#define __FPINKRED__ "\033[35m"
#define __BBLACK__ "\033[40m"
#define __BRED__ "\033[41m"
#define __BGREEN__ "\033[42m"
#define __BYELLOW__ "\033[43m"
#define __BBLUE__ "\033[44m"

#define Assert(expr)                                       \
    do                                                     \
    {                                                      \
        if (expr)                                          \
        {                                                  \
            printf(__CLEAR__                               \
                       __HIGHLIGHT__ __FRED__ #expr "\n"); \
            exit(-1);                                      \
        }                                                  \
    } while (0)

// #expr可以将expr替换为对应的字符串。