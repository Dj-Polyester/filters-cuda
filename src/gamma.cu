#include "../include/gamma.h"

// #define PRINTGAMMABENCH() PRINTCUDABENCH(MS)
#define PRINTGAMMABENCH() PRINTCUDABENCH2(MS)

__global__ void gammaAvgKernel(
    unsigned char *dstimg,
    const int cn,
    const size_t numOfPixels,
    float r, float g, float b)
{
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        float result = dstimg[icn + 0] * b +
                       dstimg[icn + 1] * g +
                       dstimg[icn + 2] * r;
        dstimg[icn + 0] = result; //B
        dstimg[icn + 1] = result; //G
        dstimg[icn + 2] = result; //R
        // pimage[icn + 3] *= result; //A
    }
}
__global__ void gammaKernel(
    unsigned char *dstimg,
    const int cn,
    const size_t numOfPixels,
    float r, float g, float b)
{
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        dstimg[icn + 0] *= b; //B
        dstimg[icn + 1] *= g; //G
        dstimg[icn + 2] *= r; //R
        // pimage[icn + 3] *= result; //A
    }
}

void gammaFilter(
    const cv::Mat &image,
    float r, float g, float b,
    void (*gammaFunc)(unsigned char *, const int, const size_t, float, float, float),
    const int blockSize)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;

    unsigned char *dstimg;

    CUDADBG(cudaMalloc(&dstimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMemcpy(dstimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    const int gridSize = (numOfPixels - 1) / blockSize + 1;
    INITCUDABENCH();

    STARTCUDABENCH();
    gammaFunc<<<gridSize, blockSize>>>(dstimg, cn, numOfPixels, r, g, b);
    CUDACHECK();
    STOPCUDABENCH();
    PRINTGAMMABENCH();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );
    CUDADBG(cudaFree(dstimg), );
}

__global__ void gammaAvgKernel2d(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        float result = dstimg[icn + 0] * b +
                       dstimg[icn + 1] * g +
                       dstimg[icn + 2] * r;
        dstimg[icn + 0] = result; //B
        dstimg[icn + 1] = result; //G
        dstimg[icn + 2] = result; //R
        // pimage[icn + 3] *= result; //A
    }
}
__global__ void gammaKernel2d(
    unsigned char *dstimg,
    const int cn,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    float r, float g, float b)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;
    unsigned icn = i * cn;

    if (i < numOfPixels)
    {
        dstimg[icn + 0] *= b; //B
        dstimg[icn + 1] *= g; //G
        dstimg[icn + 2] *= r; //R
        // pimage[icn + 3] *= result; //A
    }
}

void gammaFilter2d(
    const cv::Mat &image,
    float r, float g, float b,
    void (*gammaFunc2d)(unsigned char *, const int, const size_t, const size_t, const size_t, float, float, float),
    const int blockWidth,
    const int blockHeight)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;

    unsigned char *dstimg;

    CUDADBG(cudaMalloc(&dstimg, numOfElems * sizeof(unsigned char)), );
    CUDADBG(cudaMemcpy(dstimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / blockWidth + 1, (height - 1) / blockHeight + 1, 1);
    INITCUDABENCH();

    STARTCUDABENCH();
    gammaFunc2d<<<gridSize, blockSize>>>(dstimg, cn, width, height, numOfPixels, r, g, b);
    CUDACHECK();
    STOPCUDABENCH();
    PRINTGAMMABENCH();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );
    CUDADBG(cudaFree(dstimg), );
}