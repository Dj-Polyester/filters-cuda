#include "../include/conv_basic.h"

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
    const int cn)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
             y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = width * y + x;
    const size_t indexcn = index * cn;

    if (x < width - windowRadius && x > windowRadius - 1 && y < height - windowRadius && y > windowRadius - 1)
    {
        winType *sums = new winType[cn];
        winType *windowTotal = new winType[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = windowTotal[i] = 0;

        const size_t winSizecn = windowSize * cn;
        const size_t winWidthcn = windowWidth * cn;

        size_t icn = 0, tmp = (index - windowRadius * (width + 1)) * cn;
        while (icn < winSizecn)
        {
            for (size_t i = 0; i < windowWidth; ++i, icn += cn, tmp += cn)
            {
                for (unsigned char j = 0; j < cn; ++j)
                {
                    winType windowicnj = window[icn + j];
                    sums[j] += windowicnj * srcimg[tmp + j];
                    windowTotal[j] += windowicnj;
                }
            }
            tmp = tmp + (width * cn) - winWidthcn;
        }

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[indexcn + j] = sums[j] / windowTotal[j];
        }
        delete[] sums;
        delete[] windowTotal;
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
    cv::Mat &image,
    const Window &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const unsigned, const int),
    const int blockWidth,
    const int blockHeight)
{
    INITCUDADBG();

    cv::copyMakeBorder(image, image, window.radius, window.radius, window.radius, window.radius, cv::BORDER_CONSTANT, cv::Scalar(0));

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

    convolveFunc<<<gridSize, blockSize>>>(srcimg, dstimg, width, height, dwindow, window.width, window.size, window.elems, window.radius, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    srcimg = dstimg = NULL;
    CUDADBG(cudaFree(dwindow), );

    image = cv::Mat(image, cv::Rect(window.radius, window.radius, image.cols - 2 * window.radius, image.rows - 2 * window.radius));
}