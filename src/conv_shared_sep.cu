#include "../include/conv_shared_sep.h"

__global__ void mooreFilterSharedSep(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *windowh,
    const winType *windowv,
    const unsigned windowWidth,
    const unsigned windowRadius,
    const unsigned windowElems,
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
    if (trueBlockIdx < 0 || trueBlockIdx >= gridDim.x || trueBlockIdy < 0 || trueBlockIdy >= gridDim.y)
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
    winType *sums = new winType[cn];
    winType *windowTotal = new winType[cn];

    bool insideh = threadIdx.x < blockDim.x - windowRadius && threadIdx.x > windowRadius - 1;
    __syncthreads();
    if (insideh)
    {
        for (size_t i = 0; i < cn; ++i)
            sums[i] = windowTotal[i] = 0;
        for (size_t icn = 0, tmp = (sharedIndex - windowRadius) * cn; icn < windowElems; icn += cn, tmp += cn)
        {
            for (unsigned char j = 0; j < cn; ++j)
            {
                winType windowicnj = windowh[icn + j];
                sums[j] += windowicnj * sharedData[tmp + j];
                windowTotal[j] += windowicnj;
            }
        }
    }
    __syncthreads();
    if (insideh)
    {
        for (unsigned char j = 0; j < cn; ++j)
        {
            sharedData[sharedIndexcn + j] = sums[j] / windowTotal[j];
        }
    }
    bool insidev = threadIdx.y < blockDim.y - windowRadius && threadIdx.y > windowRadius - 1;
    if (insidev)
    {
        for (size_t i = 0; i < cn; ++i)
            sums[i] = windowTotal[i] = 0;
        size_t wcn = blockDim.x * cn;
        __syncthreads();
        for (size_t icn = 0, tmp = (sharedIndex - windowRadius * blockDim.x) * cn; icn < windowElems; icn += cn, tmp += wcn)
        {
            for (unsigned char j = 0; j < cn; ++j)
            {
                winType windowicnj = windowv[icn + j];
                sums[j] += windowicnj * sharedData[tmp + j];
                windowTotal[j] += windowicnj;
            }
        }

        for (unsigned char j = 0; j < cn; ++j)
        {
            dstimg[imgIndexcn + j] = sums[j] / windowTotal[j];
        }
    }
    delete[] sums;
    delete[] windowTotal;
}

void convolveSharedSep(
    cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFunc)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const winType *, const unsigned, const unsigned, const unsigned, const int),
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
    winType *dwindowh = NULL, *dwindowv = NULL;

    CUDADBG(cudaMalloc(&dwindowh, 2 * (window.elems * sizeof(winType) + numOfElems * sizeof(unsigned char))), );
    dwindowv = dwindowh + window.elems;
    srcimg = (unsigned char *)(dwindowv + window.elems);
    dstimg = srcimg + numOfElems;

    CUDADBG(cudaMemcpy(srcimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindowh, window.hdata, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dwindowv, window.vdata, window.elems * sizeof(winType), cudaMemcpyHostToDevice), );

    unsigned trueBlockWidth = blockWidth - 2 * window.radius, trueBlockHeight = blockHeight - 2 * window.radius;
    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / trueBlockWidth + 1, (height - 1) / trueBlockHeight + 1, 1);
    const size_t sharedSize = blockWidth * blockHeight * cn;

    convolveFunc<<<gridSize, blockSize, sharedSize>>>(srcimg, dstimg, width, height, dwindowh, dwindowv, window.width, window.radius, window.elems, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    srcimg = dstimg = NULL;
    dwindowv = NULL;
    CUDADBG(cudaFree(dwindowh), );
}