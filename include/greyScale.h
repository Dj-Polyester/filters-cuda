#if !defined(GREYSCALE_H)
#define GREYSCALE_H

#include "global.h"

struct greyScaleArgs
{
    float gamma;
    unsigned char lowerbound, upperbound;
};

__global__ void binarize(
    unsigned char *dstimg,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const greyScaleArgs *dargs);

__global__ void negate(
    unsigned char *dstimg,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const greyScaleArgs *dargs);

__global__ void powerLaw(
    unsigned char *dstimg,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const greyScaleArgs *dargs);

void greyScaleTransform(
    const cv::Mat &image,
    const greyScaleArgs &args,
    void (*greyScaleTransformFunc)(unsigned char *, const size_t, const size_t, const size_t, const greyScaleArgs *),
    const int blockWidth,
    const int blockHeight);

#endif // GREYSCALE_H
