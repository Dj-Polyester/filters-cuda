#if !defined(CONVOLUTION)
#define CONVOLUTION

#include "global.h"

typedef float winType;

enum WindowType
{
    mean
};
struct SeparableWindow
{
    winType *hdata, *vdata;
    const size_t width, elems;
    const unsigned channels;
    SeparableWindow(const size_t &w, const unsigned &cn, WindowType type) : channels(cn), width(w), elems(cn * w)
    {
        cudaMallocHost(&hdata, 2 * elems * sizeof(winType));
        vdata = hdata + elems;
        switch (type)
        {
        case mean:
            for (size_t i = 0; i < elems; ++i)
            {
                hdata[i] = vdata[i] = 1;
            }
            break;

        default:
            break;
        }
    }
    ~SeparableWindow()
    {
        vdata = NULL;
        cudaFreeHost(hdata);
    }
};
struct Window
{
    winType *data;
    const size_t width, size, elems;
    const unsigned channels;
    Window(const size_t &w, const unsigned &cn, WindowType type) : channels(cn), width(w), size(w * w), elems(cn * w * w)
    {
        cudaMallocHost(&data, elems * sizeof(winType));
        switch (type)
        {
        case mean:
            for (size_t i = 0; i < elems; ++i)
            {
                data[i] = 1;
            }
            break;

        default:
            break;
        }
    }
    ~Window()
    {
        cudaFreeHost(data);
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

__global__ void mooreFilter(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowSize,
    const unsigned windowElems,
    const int cn);

void convolve(
    const cv::Mat &image,
    const Window &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const int),
    const int blockSize);

__global__ void mooreFilter2dSeparableH(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn);

__global__ void mooreFilter2dSeparableV(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn);

void convolve2dSeparable(
    const cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFunc2dH)(const unsigned char *, unsigned char *, const size_t, const winType *, const unsigned, const unsigned, const int),
    void (*convolveFunc2dV)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight);

__global__ void mooreFilterSeparableH(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn);

__global__ void mooreFilterSeparableV(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn);

void convolveSeparable(
    const cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFunc2dH)(const unsigned char *, unsigned char *, const size_t, const winType *, const unsigned, const unsigned, const int),
    void (*convolveFunc2dV)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const int),
    const int blockSize);

#endif // CONVOLUTION
