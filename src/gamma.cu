#include "../include/gamma.h"

// #define PRINTGAMMABENCH() PRINTCUDABENCH(MS)
#define PRINTGAMMABENCH() PRINTCUDABENCH2(MS)

__global__ void gammaAvgKernel(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t numOfPixels,
    float r, float g, float b, float a)
{
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        float result = 0;
        switch (howmany)
        {
        case 4:
            result += dstimg[icn + 3] * a;
        case 3:
            result += dstimg[icn + 0] * b;
        case 2:
            result += dstimg[icn + 1] * g;
        case 1:
            result += dstimg[icn + 2] * r;
        }
        switch (howmany)
        {
        case 4:
            dstimg[icn + 3] = result;
        case 3:
            dstimg[icn + 0] = result;
        case 2:
            dstimg[icn + 1] = result;
        case 1:
            dstimg[icn + 2] = result;
        }
    }
}
__global__ void gammaKernel(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t numOfPixels,
    float r, float g, float b, float a)
{
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        switch (howmany)
        {
        case 4:
            dstimg[icn + 3] *= a;
        case 3:
            dstimg[icn + 0] *= b;
        case 2:
            dstimg[icn + 1] *= g;
        case 1:
            dstimg[icn + 2] *= r;
        }
    }
}

void gammaFilter(
    const cv::Mat &image, const unsigned char howmany,
    float r, float g, float b, float a,
    void (*gammaFunc)(unsigned char *, const int, const unsigned char, const size_t, float, float, float, float),
    const int blockSize)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;

    if (howmany > cn)
        ERROR("gamma size greater than number of channels.", )

    unsigned char *dstimg;

    CUDADBG(cudaMalloc(&dstimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMemcpy(dstimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    const int gridSize = (numOfPixels - 1) / blockSize + 1;
    INITCUDABENCH();

    STARTCUDABENCH();
    gammaFunc<<<gridSize, blockSize>>>(dstimg, cn, howmany, numOfPixels, r, g, b, a);
    CUDACHECK();
    STOPCUDABENCH();
    PRINTGAMMABENCH();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );
    CUDADBG(cudaFree(dstimg), );
}

__global__ void gammaAvgKernel2d(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b, float a)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        float result = 0;
        switch (howmany)
        {
        case 4:
            result += dstimg[icn + 3] * a;
        case 3:
            result += dstimg[icn + 0] * b;
        case 2:
            result += dstimg[icn + 1] * g;
        case 1:
            result += dstimg[icn + 2] * r;
        }
        switch (howmany)
        {
        case 4:
            dstimg[icn + 3] = result;
        case 3:
            dstimg[icn + 0] = result;
        case 2:
            dstimg[icn + 1] = result;
        case 1:
            dstimg[icn + 2] = result;
        }
    }
}
__global__ void gammaKernel2d(
    unsigned char *dstimg,
    const int cn, const unsigned char howmany,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b, float a)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        switch (howmany)
        {
        case 4:
            dstimg[icn + 3] *= a;
        case 3:
            dstimg[icn + 0] *= b;
        case 2:
            dstimg[icn + 1] *= g;
        case 1:
            dstimg[icn + 2] *= r;
        }
    }
}

void gammaFilter2d(
    const cv::Mat &image, const unsigned char howmany,
    float r, float g, float b, float a,
    void (*gammaFunc2d)(unsigned char *, const int, const unsigned char, const size_t, const size_t, const size_t, float, float, float, float),
    const int blockWidth,
    const int blockHeight)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    if (howmany > cn)
    {
        ERROR("gamma size greater than number of channels.", )
    }

    const int height = image.rows;

    unsigned char *dstimg;

    CUDADBG(cudaMalloc(&dstimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMemcpy(dstimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / blockWidth + 1, (height - 1) / blockHeight + 1, 1);
    INITCUDABENCH();

    STARTCUDABENCH();
    gammaFunc2d<<<gridSize, blockSize>>>(dstimg, cn, howmany, width, height, numOfPixels, r, g, b, a);
    CUDACHECK();
    STOPCUDABENCH();
    PRINTGAMMABENCH();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );
    CUDADBG(cudaFree(dstimg), );
}