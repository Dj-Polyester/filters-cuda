#if !defined(SAMPLING_H)
#define SAMPLING_H

#include "global.h"

__global__ void NN(unsigned char *srcimg, unsigned char *dstimg, int cn, size_t width, size_t height, unsigned factorx, unsigned factory);

__global__ void bilinear(unsigned char *srcimg, unsigned char *dstimg, int cn, size_t width, size_t newwidth, unsigned factorx, unsigned factory);

void interpolate(cv::Mat &image,
                 void (*interpolateFunc)(unsigned char *srcimg, unsigned char *dstimg, int cn, size_t width, size_t height, unsigned factorx, unsigned factory),
                 const int blockWidth,
                 const int blockHeight,
                 unsigned factorx,
                 unsigned factory);

#endif // SAMPLING_H
