#if !defined(SEPARABLE)
#define SEPARABLE

#include "window.h"

struct SeparableWindow
{
    winType *hdata, *vdata;
    size_t radius, width, elems;
    const unsigned channels;
    SeparableWindow(const size_t &r, const unsigned &cn, WindowType type, const winType doublevar = 2) : channels(cn), radius(r)
    {

        width = 2 * r + 1;
        elems = cn * width;

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
        case gaussian:
            for (size_t i = 0; i < width; ++i)
            {
                long disti = i - radius;
                winType gaussianTmp = GAUSSIAN(disti, doublevar);
                for (size_t j = 0; j < cn; ++j)
                {
                    size_t tmpindex = i * cn + j;
                    hdata[tmpindex] = vdata[tmpindex] = gaussianTmp;
                }
            }
            break;

        default:
            break;
        }
    }
    SeparableWindow(std::vector<winType> &&winH, std::vector<winType> &&winV, const unsigned &cn) : channels(cn)
    {
        width = winH.size();
        radius = width / 2;
        elems = width * cn;
        cudaMallocHost(&hdata, 2 * elems * sizeof(winType));
        vdata = hdata + elems;
        for (size_t i = 0; i < width; ++i)
        {
            winType windowiH = winH[i];
            winType windowiV = winV[i];
            for (size_t j = 0; j < cn; ++j)
            {
                hdata[i * cn + j] = windowiH;
                vdata[i * cn + j] = windowiV;
            }
        }
    }
    ~SeparableWindow()
    {
        vdata = NULL;
        cudaFreeHost(hdata);
    }
};

__global__ void mooreFilterSeparableH(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const winType *window,
    const unsigned windowRadius,
    const unsigned windowElems,
    const int cn);

__global__ void mooreFilterSeparableV(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowRadius,
    const unsigned windowElems,
    const int cn);

void convolveSeparable(
    cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFuncH)(const unsigned char *, unsigned char *, const size_t, const winType *, const unsigned, const unsigned, const int),
    void (*convolveFuncV)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight);

#endif // SEPARABLE
