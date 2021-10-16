#include "../include/conv_shared.h"

__global__ void mooreFilterShared(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowWidth,
    const unsigned windowSize,
    const unsigned windowRadius,
    const int cn)
{
    // dynamic shared memory
    extern __shared__ unsigned char sharedData[];
    // get true dimensions
    int thresholdx = blockDim.x - windowRadius,
        thresholdy = blockDim.y - windowRadius,
        trueBlockDimx = thresholdx - windowRadius,
        trueBlockDimy = thresholdy - windowRadius;

    int distx, disty,
        trueBlockIdx, trueBlockIdy,
        trueThreadIdx, trueThreadIdy;

    if (threadIdx.x < windowRadius)
    {
        distx = windowRadius - threadIdx.x - 1;
        trueBlockIdx = blockIdx.x - distx / trueBlockDimx - 1;
        trueThreadIdx = trueBlockDimx - (distx % trueBlockDimx) - 1;
    }
    else if (threadIdx.x >= thresholdx)
    {
        distx = threadIdx.x - thresholdx;
        trueBlockIdx = blockIdx.x + distx / trueBlockDimx + 1;
        trueThreadIdx = distx % trueBlockDimx;
    }
    else
    {
        trueBlockIdx = blockIdx.x;
        trueThreadIdx = threadIdx.x - windowRadius;
    }

    if (threadIdx.y < windowRadius)
    {
        disty = windowRadius - threadIdx.y - 1;
        trueBlockIdy = blockIdx.y - disty / trueBlockDimy - 1;
        trueThreadIdy = trueBlockDimy - (disty % trueBlockDimy) - 1;
    }
    else if (threadIdx.y >= thresholdy)
    {
        disty = threadIdx.y - thresholdy;
        trueBlockIdy = blockIdx.y + disty / trueBlockDimy + 1;
        trueThreadIdy = disty % trueBlockDimy;
    }
    else
    {
        trueBlockIdy = blockIdx.y;
        trueThreadIdy = threadIdx.y - windowRadius;
    }

    const unsigned sharedIndex = blockDim.x * threadIdx.y + threadIdx.x;
    const unsigned sharedIndexcn = sharedIndex * cn;
    // apron pixels
    if (trueBlockIdx < 0 || trueBlockIdy < 0 || trueBlockIdx >= gridDim.x || trueBlockIdy >= gridDim.y)
    {
        for (size_t i = 0; i < cn; ++i)
        {
            sharedData[sharedIndexcn + i] = 0;
        }
        return;
    }

    const unsigned imgx = trueBlockDimx * trueBlockIdx + trueThreadIdx,
                   imgy = trueBlockDimy * trueBlockIdy + trueThreadIdy;

    const unsigned imgIndex = imgy * width + imgx;
    const unsigned imgIndexcn = imgIndex * cn;
    // load shared memory
    for (size_t i = 0; i < cn; ++i)
    {
        sharedData[sharedIndexcn + i] = srcimg[imgIndexcn + i];
    }
    // convolve if inside
    if (threadIdx.x < blockDim.x - windowRadius && threadIdx.x > windowRadius - 1 && threadIdx.y < blockDim.y - windowRadius && threadIdx.y > windowRadius - 1)
    {
        winType *sums = new winType[cn];
        winType *windowTotal = new winType[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = windowTotal[i] = 0;

        const size_t winSizecn = windowSize * cn;
        const size_t winWidthcn = windowWidth * cn;
        __syncthreads();
        size_t icn = 0, tmp = (sharedIndex - windowRadius * (blockDim.x + 1)) * cn;
        while (icn < winSizecn)
        {
            for (size_t i = 0; i < windowWidth; ++i, icn += cn, tmp += cn)
            {
                for (unsigned char j = 0; j < cn; ++j)
                {
                    winType windowicnj = window[icn + j];
                    sums[j] += windowicnj * sharedData[tmp + j];
                    windowTotal[j] += windowicnj;
                }
            }
            tmp = tmp + (blockDim.x * cn) - winWidthcn;
        }

        // load global memory back
        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[imgIndexcn + j] = sums[j] / windowTotal[j];
        }
        delete[] sums;
        delete[] windowTotal;
    }
}

void convolveShared(
    cv::Mat &image,
    const Window &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const unsigned, const int),
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

    unsigned trueBlockWidth = blockWidth - 2 * window.radius, trueBlockHeight = blockHeight - 2 * window.radius;
    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / trueBlockWidth + 1, (height - 1) / trueBlockHeight + 1, 1);
    const size_t sharedSize = blockWidth * blockHeight * cn;

    convolveFunc<<<gridSize, blockSize, sharedSize>>>(srcimg, dstimg, width, height, dwindow, window.width, window.size, window.radius, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    srcimg = dstimg = NULL;
    CUDADBG(cudaFree(dwindow), );
}