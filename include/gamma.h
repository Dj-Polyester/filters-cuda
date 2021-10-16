#if !defined(CUDAGAMMA)
#define CUDAGAMMA

#include "global.h"

#define greyScale(img, blockWidth, blockHeight) gammaFilter(img, {.114, .587, .299}, gammaAvgKernel, blockWidth, blockHeight)

__global__ void gammaAvgKernel(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const float *gammaVals,
    const size_t howmany);

__global__ void gammaKernel(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const float *gammaVals,
    const size_t howmany);

void gammaFilter(
    const cv::Mat &image,
    std::vector<float> gammaVals, void (*gammaFunc)(unsigned char *, const int, const size_t, const size_t, const size_t, const float *, const size_t),
    const int blockWidth,
    const int blockHeight);

#endif // CUDAGAMMA
