#if !defined(BASIC_CONV)
#define BASIC_CONV

#include "window.h"

struct Window
{
    winType *data;
    size_t width, size, elems, radius;
    const unsigned channels;
    Window(const size_t &r, const unsigned &cn, WindowType type, const winType doublevar = 2) : channels(cn), radius(r)
    {
        width = 2 * r + 1;
        size = width * width;
        elems = cn * size;
        cudaMallocHost(&data, elems * sizeof(winType));
        switch (type)
        {
        case mean:
            for (size_t i = 0; i < elems; ++i)
            {
                data[i] = 1;
            }
            break;
        case gaussian:
            for (size_t i = 0; i < size; ++i)
            {
                long disti = i % width - radius;
                long distj = i / width - radius;
                winType gaussianTmp = GAUSSIAN2D(disti, distj, doublevar);
                for (size_t j = 0; j < cn; ++j)
                {
                    data[i * cn + j] = gaussianTmp;
                }
            }
            break;
        default:
            break;
        }
    }
    Window(std::vector<winType> &&window, const unsigned &cn) : channels(cn)
    {
        size = window.size();
        width = sqrt(size);
        elems = size * cn;
        radius = width / 2;
        cudaMallocHost(&data, elems * sizeof(winType));
        for (size_t i = 0; i < size; ++i)
        {
            winType windowi = window[i];
            for (size_t j = 0; j < cn; ++j)
            {
                data[i * cn + j] = windowi;
            }
        }
    }
    ~Window()
    {
        cudaFreeHost(data);
    }
};

__global__ void mooreFilter(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowSize,
    const unsigned windowElems,
    const unsigned windowRadius,
    const int cn);

void convolve(
    cv::Mat &image,
    const Window &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight);

#endif // BASIC_CONV
