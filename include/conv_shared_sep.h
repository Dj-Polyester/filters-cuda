#if !defined(CONV_SHARED_SEP)
#define CONV_SHARED_SEP

#include "conv_separable.h"

__global__ void mooreFilterSharedSep(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *windowh,
    const winType *windowv,
    const unsigned windowWidth,
    const unsigned windowRadius,
    const unsigned windowElems,
    const int cn);

void convolveSharedSep(
    cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const winType *, const unsigned, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight);

#endif // CONV_SHARED_SEP
