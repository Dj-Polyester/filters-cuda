#include "../include/convolution.h"

#define PRINTCONVOLUTIONBENCH() PRINTCUDABENCH2(MS)

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

    unsigned char *srcimg, *dstimg;
    winType *dwindow;

    CUDADBG(cudaMalloc(&dstimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMalloc(&srcimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMemcpy(srcimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    CUDADBG(cudaMalloc(&dwindow, window.elems * sizeof(winType)), );
    CUDADBG(cudaMemcpy(dwindow, window.data.data(), window.elems * sizeof(winType), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / blockWidth + 1, (height - 1) / blockHeight + 1, 1);
    INITCUDABENCH();

    STARTCUDABENCH();
    convolveFunc2d<<<gridSize, blockSize>>>(srcimg, dstimg, width, height, dwindow, window.width, window.size, window.elems, cn);
    CUDACHECK();
    STOPCUDABENCH();
    PRINTCONVOLUTIONBENCH();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    CUDADBG(cudaFree(dstimg), );
    CUDADBG(cudaFree(dwindow), );
}