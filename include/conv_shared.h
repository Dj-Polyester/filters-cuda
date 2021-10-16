#if !defined(CONV_SHARED)
#define CONV_SHARED

#include "conv_basic.h"

__global__ void mooreFilterShared(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowSize,
    const unsigned windowRadius,
    const int cn);

void convolveShared(
    cv::Mat &image,
    const Window &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight);

#endif // CONV_SHARED
