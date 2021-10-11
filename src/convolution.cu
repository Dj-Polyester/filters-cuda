#include "../include/convolution.h"

#define PRINTCONVOLUTIONBENCH() PRINTCUDABENCH2(MS)

__global__ void mooreFilter(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowSize,
    const unsigned windowElems,
    const int cn)
{
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t indexcn = index * cn;
    const size_t x = index % width,
                 y = index / width;
    unsigned ndiv2 = windowWidth / 2;
    if (x < width - ndiv2 && x > ndiv2 - 1 && y < height - ndiv2 && y > ndiv2 - 1)
    {
        unsigned *sums = new unsigned[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = 0;

        const size_t winSizecn = windowSize * cn;
        const size_t winWidthcn = windowWidth * cn;

        size_t icn = 0, tmp = (index - (windowWidth / 2) * (width + 1)) * cn;

        while (icn < winSizecn)
        {
            for (size_t i = 0; i < windowWidth; ++i, icn += cn, tmp += cn)
            {
                // printf("(%u,%u) index: %lu, cn: %u, indexcn: %lu, icn: %lu, tmp: %lu, winSizecn: %lu\n", x, y, index, cn, indexcn, icn, tmp, winSizecn);
                for (unsigned char j = 0; j < cn; ++j)
                    sums[j] += window[icn + j] * srcimg[tmp + j];
            }
            // printf("(%u,%u) loop1 out\n", x, y);
            tmp = tmp + (width * cn) - winWidthcn;
        }
        // printf("(%u,%u) loop2 out\n", x, y);

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = sums[j] / windowSize;
        }
        delete[] sums;
    }
    else
    {
        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = srcimg[indexcn + j];
        }
    }
    // printf("(%u,%u) (%u,%u,%u) (%u,%u,%u)\n", x, y, srcimg[indexcn + 0], srcimg[indexcn + 1], srcimg[indexcn + 2], dstimg[indexcn + 0], dstimg[indexcn + 1], dstimg[indexcn + 2]);
}

__global__ void mooreFilterSeparableH(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn)
{
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t indexcn = index * cn;
    const size_t x = index % width;

    unsigned ndiv2 = windowWidth / 2;
    if (x < width - ndiv2 && x > ndiv2 - 1)
    {
        unsigned *sums = new unsigned[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = 0;

        for (size_t icn = 0, tmp = (index - (windowWidth / 2)) * cn; icn < windowElems; icn += cn, tmp += cn)
        {
            for (unsigned char j = 0; j < cn; ++j)
                sums[j] += window[icn + j] * srcimg[tmp + j];
        }

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = sums[j] / windowWidth;
        }
        delete[] sums;
    }
    else
    {
        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = srcimg[indexcn + j];
        }
    }
}
__global__ void mooreFilterSeparableV(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn)
{
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t indexcn = index * cn;
    const size_t y = index / width;

    unsigned ndiv2 = windowWidth / 2;
    if (y < height - ndiv2 && y > ndiv2 - 1)
    {
        unsigned *sums = new unsigned[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = 0;
        size_t wcn = width * cn;
        for (size_t icn = 0, tmp = (index - (windowWidth / 2) * width) * cn; icn < windowElems; icn += cn, tmp += wcn)
        {
            for (unsigned char j = 0; j < cn; ++j)
                sums[j] += window[icn + j] * srcimg[tmp + j];
        }

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = sums[j] / windowWidth;
        }
        delete[] sums;
    }
    else
    {
        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = srcimg[indexcn + j];
        }
    }
}

__global__ void mooreFilter2dSeparableH(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
             y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = width * y + x;
    const size_t indexcn = index * cn;

    unsigned ndiv2 = windowWidth / 2;
    if (x < width - ndiv2 && x > ndiv2 - 1)
    {
        unsigned *sums = new unsigned[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = 0;

        for (size_t icn = 0, tmp = (index - (windowWidth / 2)) * cn; icn < windowElems; icn += cn, tmp += cn)
        {
            for (unsigned char j = 0; j < cn; ++j)
                sums[j] += window[icn + j] * srcimg[tmp + j];
        }

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = sums[j] / windowWidth;
        }
        delete[] sums;
    }
    else
    {
        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = srcimg[indexcn + j];
        }
    }
}
__global__ void mooreFilter2dSeparableV(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowElems,
    const int cn)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
             y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = width * y + x;
    const size_t indexcn = index * cn;

    unsigned ndiv2 = windowWidth / 2;
    if (y < height - ndiv2 && y > ndiv2 - 1)
    {
        unsigned *sums = new unsigned[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = 0;
        size_t wcn = width * cn;
        for (size_t icn = 0, tmp = (index - (windowWidth / 2) * width) * cn; icn < windowElems; icn += cn, tmp += wcn)
        {
            for (unsigned char j = 0; j < cn; ++j)
                sums[j] += window[icn + j] * srcimg[tmp + j];
        }

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = sums[j] / windowWidth;
        }
        delete[] sums;
    }
    else
    {
        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = srcimg[indexcn + j];
        }
    }
}

void convolve(
    const cv::Mat &image,
    const Window &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const int),
    const int blockSize)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;

    unsigned char *srcimg, *dstimg;
    winType *dwindow;

    CUDADBG(cudaMalloc(&dwindow, window.elems * sizeof(winType) + 2 * numOfElems * sizeof(unsigned char)), );
    dstimg = (unsigned char *)(dwindow + window.elems);
    srcimg = dstimg + numOfElems;
    CUDADBG(cudaMemcpy(srcimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindow, window.data, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );

    const int gridSize = (numOfPixels - 1) / blockSize + 1;

    convolveFunc<<<gridSize, blockSize>>>(srcimg, dstimg, width, height, dwindow, window.width, window.size, window.elems, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    dstimg = NULL;
    srcimg = NULL;
    CUDADBG(cudaFree(dwindow), );
}

__global__ void mooreFilter2d(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowSize,
    const unsigned windowElems,
    const int cn)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
             y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = width * y + x;
    const size_t indexcn = index * cn;

    unsigned ndiv2 = windowWidth / 2;
    if (x < width - ndiv2 && x > ndiv2 - 1 && y < height - ndiv2 && y > ndiv2 - 1)
    {
        unsigned *sums = new unsigned[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = 0;

        const size_t winSizecn = windowSize * cn;
        const size_t winWidthcn = windowWidth * cn;

        size_t icn = 0, tmp = (index - (windowWidth / 2) * (width + 1)) * cn;

        while (icn < winSizecn)
        {
            for (size_t i = 0; i < windowWidth; ++i, icn += cn, tmp += cn)
            {
                // printf("(%u,%u) index: %lu, cn: %u, indexcn: %lu, icn: %lu, tmp: %lu, winSizecn: %lu\n", x, y, index, cn, indexcn, icn, tmp, winSizecn);
                for (unsigned char j = 0; j < cn; ++j)
                    sums[j] += window[icn + j] * srcimg[tmp + j];
            }
            // printf("(%u,%u) loop1 out\n", x, y);
            tmp = tmp + (width * cn) - winWidthcn;
        }
        // printf("(%u,%u) loop2 out\n", x, y);

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = sums[j] / windowSize;
        }
        delete[] sums;
    }
    else
    {
        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = srcimg[indexcn + j];
        }
    }
    // printf("(%u,%u) (%u,%u,%u) (%u,%u,%u)\n", x, y, srcimg[indexcn + 0], srcimg[indexcn + 1], srcimg[indexcn + 2], dstimg[indexcn + 0], dstimg[indexcn + 1], dstimg[indexcn + 2]);
}

void convolve2d(
    const cv::Mat &image,
    const Window &window,
    void (*convolveFunc2d)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;

    unsigned char *srcimg = NULL, *dstimg = NULL;
    winType *dwindow = NULL;

    CUDADBG(cudaMalloc(&dwindow, window.elems * sizeof(winType) + 2 * numOfElems * sizeof(unsigned char)), );
    srcimg = (unsigned char *)(dwindow + window.elems);
    dstimg = srcimg + numOfElems;
    CUDADBG(cudaMemcpy(srcimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindow, window.data, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / blockWidth + 1, (height - 1) / blockHeight + 1, 1);

    convolveFunc2d<<<gridSize, blockSize>>>(srcimg, dstimg, width, height, dwindow, window.width, window.size, window.elems, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    srcimg = dstimg = NULL;
    CUDADBG(cudaFree(dwindow), );
}
void convolve2dSeparable(
    const cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFunc2dH)(const unsigned char *, unsigned char *, const size_t, const winType *, const unsigned, const unsigned, const int),
    void (*convolveFunc2dV)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;

    unsigned char *srcimg = NULL, *dstimg = NULL, *interimg = NULL;
    winType *dwindowh = NULL, *dwindowv = NULL;

    CUDADBG(cudaMalloc(&dwindowh, 2 * window.elems * sizeof(winType) + 3 * numOfElems * sizeof(unsigned char)), );
    dwindowv = dwindowh + window.elems;
    srcimg = (unsigned char *)(dwindowv + window.elems);
    interimg = srcimg + numOfElems;
    dstimg = interimg + numOfElems;

    CUDADBG(cudaMemcpy(srcimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindowh, window.hdata, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindowv, window.vdata, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / blockWidth + 1, (height - 1) / blockHeight + 1, 1);

    convolveFunc2dH<<<gridSize, blockSize>>>(srcimg, interimg, width, dwindowh, window.width, window.elems, cn);
    CUDACHECK();
    convolveFunc2dV<<<gridSize, blockSize>>>(interimg, dstimg, width, height, dwindowv, window.width, window.elems, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    srcimg = dstimg = interimg = NULL;
    dwindowv = NULL;
    CUDADBG(cudaFree(dwindowh), );
}
void convolveSeparable(
    const cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFunc2dH)(const unsigned char *, unsigned char *, const size_t, const winType *, const unsigned, const unsigned, const int),
    void (*convolveFunc2dV)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const int),
    const int blockSize)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;

    unsigned char *srcimg = NULL, *dstimg = NULL, *interimg = NULL;
    winType *dwindowh = NULL, *dwindowv = NULL;

    CUDADBG(cudaMalloc(&dwindowh, 2 * window.elems * sizeof(winType) + 3 * numOfElems * sizeof(unsigned char)), );
    dwindowv = dwindowh + window.elems;
    srcimg = (unsigned char *)(dwindowv + window.elems);
    interimg = srcimg + numOfElems;
    dstimg = interimg + numOfElems;

    CUDADBG(cudaMemcpy(srcimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindowh, window.hdata, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindowv, window.vdata, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );

    const int gridSize = (numOfPixels - 1) / blockSize + 1;

    convolveFunc2dH<<<gridSize, blockSize>>>(srcimg, interimg, width, dwindowh, window.width, window.elems, cn);
    CUDACHECK();
    convolveFunc2dV<<<gridSize, blockSize>>>(interimg, dstimg, width, height, dwindowv, window.width, window.elems, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    srcimg = dstimg = interimg = NULL;
    dwindowv = NULL;
    CUDADBG(cudaFree(dwindowh), );
}