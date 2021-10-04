#if !defined(CUDAGAMMA)
#define CUDAGAMMA

#include "global.h"

#define greyScale(img, blockSize) gammaFilter(img, 3, .114, .587, .299, 0, gammaAvgKernel, blockSize)
#define greyScale2d(img, blockWidth, blockHeight) gammaFilter2d(img, 3, .114, .587, .299, 0, gammaAvgKernel2d, blockWidth, blockHeight)

__global__ void gammaAvgKernel(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t numOfPixels,
    float r, float g, float b, float a);

__global__ void gammaKernel(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t numOfPixels,
    float r, float g, float b, float a);

void gammaFilter(
    const cv::Mat &image, const unsigned char howmany,
    float r, float g, float b, float a,
    void (*gammaFunc)(unsigned char *, const int, const unsigned char, const size_t, float, float, float, float),
    const int blockSize);

__global__ void gammaAvgKernel2d(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b, float a);

__global__ void gammaKernel2d(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b, float a);

void gammaFilter2d(
    const cv::Mat &image, const unsigned char howmany,
    float r, float g, float b, float a,
    void (*gammaFunc2d)(unsigned char *, const int, const unsigned char, const size_t, const size_t, const size_t, float, float, float, float),
    const int blockWidth,
    const int blockHeight);

#endif // CUDAGAMMA
