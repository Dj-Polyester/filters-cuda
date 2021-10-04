#include "../include/gamma.h"

// #define PRINTGAMMABENCH() PRINTCUDABENCH(MS)
#define PRINTGAMMABENCH() PRINTCUDABENCH2(MS)

__global__ void gammaAvgKernel(
    unsigned char *dstimg,
    const int cn,
    const size_t numOfPixels,
    const float *gammaVals,
    const size_t howmany)
{
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        float result = 0;
        for (size_t j = 0; j < howmany; ++j)
        {
            result += dstimg[icn + j] * gammaVals[j];
        }
        for (size_t j = 0; j < howmany; ++j)
        {
            dstimg[icn + j] = result;
        }
    }
}
__global__ void gammaKernel(
    unsigned char *dstimg,
    const int cn,
    const size_t numOfPixels,
    const float *gammaVals,
    const size_t howmany)
{
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        for (size_t j = 0; j < howmany; ++j)
        {
            dstimg[icn + j] *= gammaVals[j];
        }
    }
}

void gammaFilter(
    const cv::Mat &image,
    std::vector<float> gammaVals,
    void (*gammaFunc)(unsigned char *, const int, const size_t, const float *, const size_t),
    const int blockSize)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    size_t howmany = gammaVals.size();
    if (howmany > cn)
        ERROR("gamma size greater than number of channels.", )

    unsigned char *dstimg;
    float *gammaValsPtr;

    CUDADBG(cudaMalloc(&dstimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMemcpy(dstimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    CUDADBG(cudaMalloc(&gammaValsPtr, howmany * sizeof(float)), );
    CUDADBG(cudaMemcpy(gammaValsPtr, gammaVals.data(), howmany * sizeof(float), cudaMemcpyHostToDevice), );

    const int gridSize = (numOfPixels - 1) / blockSize + 1;

    gammaFunc<<<gridSize, blockSize>>>(dstimg, cn, numOfPixels, gammaValsPtr, howmany);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );
    CUDADBG(cudaFree(dstimg), );
    CUDADBG(cudaFree(gammaValsPtr), );
}

__global__ void gammaAvgKernel2d(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const float *gammaVals,
    const size_t howmany)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        float result = 0;
        for (size_t j = 0; j < howmany; ++j)
        {
            result += dstimg[icn + j] * gammaVals[j];
        }
        for (size_t j = 0; j < howmany; ++j)
        {
            dstimg[icn + j] = result;
        }
    }
}
__global__ void gammaKernel2d(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const float *gammaVals,
    const size_t howmany)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        for (size_t j = 0; j < howmany; ++j)
        {
            dstimg[icn + j] *= gammaVals[j];
        }
    }
}

void gammaFilter2d(
    const cv::Mat &image,
    std::vector<float> gammaVals, void (*gammaFunc2d)(unsigned char *, const int, const size_t, const size_t, const size_t, const float *, const size_t),
    const int blockWidth,
    const int blockHeight)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;

    size_t howmany = gammaVals.size();

    if (howmany > cn)
    {
        ERROR("gamma size greater than number of channels.", )
    }

    unsigned char *dstimg;
    float *gammaValsPtr;

    CUDADBG(cudaMalloc(&dstimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMemcpy(dstimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    CUDADBG(cudaMalloc(&gammaValsPtr, howmany * sizeof(float)), );
    CUDADBG(cudaMemcpy(gammaValsPtr, gammaVals.data(), howmany * sizeof(float), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / blockWidth + 1, (height - 1) / blockHeight + 1, 1);

    gammaFunc2d<<<gridSize, blockSize>>>(dstimg, cn, width, height, numOfPixels, gammaValsPtr, howmany);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );
    CUDADBG(cudaFree(dstimg), );
}