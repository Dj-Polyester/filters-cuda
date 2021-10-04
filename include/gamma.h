#if !defined(CUDAGAMMA)
#define CUDAGAMMA

#include "global.h"

#define greyScale(img, blockSize) gammaFilter(img, .114, .587, .299, gammaAvgKernel, blockSize)
#define greyScale2d(img, blockWidth, blockHeight) gammaFilter2d(img, .114, .587, .299, gammaAvgKernel2d, blockWidth, blockHeight)

__global__ void gammaAvgKernel(
    unsigned char *dimage,
    const int cn,
    const size_t numOfPixels, float r, float g, float b);

__global__ void gammaKernel(
    unsigned char *dimage,
    const int cn,
    const size_t numOfPixels, float r, float g, float b);

void gammaFilter(
    const cv::Mat &image,
    float r, float g, float b,
    void (*gammaFunc)(unsigned char *, const int, const size_t, float, float, float),
    const int blockSize);

__global__ void gammaAvgKernel2d(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b);

__global__ void gammaKernel2d(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b);

void gammaFilter2d(
    const cv::Mat &image,
    float r, float g, float b,
    void (*gammaFunc2d)(unsigned char *, const int, const size_t, const size_t, const size_t, float, float, float),
    const int blockWidth,
    const int blockHeight);

#endif // CUDAGAMMA
