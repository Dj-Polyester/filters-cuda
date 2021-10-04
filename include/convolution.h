#if !defined(CONVOLUTION)
#define CONVOLUTION

#include "global.h"

typedef float winType;

enum WindowType
{
    mean
};
struct Window
{
    std::vector<winType> data;
    const size_t width, size, elems;
    const unsigned channels;
    Window(const size_t &w, const unsigned &cn, WindowType type) : channels(cn), width(w), size(w * w), elems(cn * w * w)
    {
        switch (type)
        {
        case mean:
            data.resize(elems, 1);
            break;

        default:
            break;
        }
    }
};

__global__ void mooreFilter2d(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowSize,
    const unsigned windowElems,
    const int cn);

void convolve2d(
    const cv::Mat &image,
    const Window &window,
    void (*convolveFunc2d)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight);

#endif // CONVOLUTION
