#include "../include/greyScale.h"

__global__ void binarize(
    unsigned char *dstimg,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const greyScaleArgs *dargs)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;

    if (i < numOfPixels)
    {
        dstimg[i] = (dstimg[i] < dargs->lowerbound) ? 0 : 255;
    }
}
__global__ void negate(
    unsigned char *dstimg,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const greyScaleArgs *dargs)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;

    if (i < numOfPixels)
    {
        dstimg[i] = 255 - dstimg[i];
    }
}
__global__ void powerLaw(
    unsigned char *dstimg,
    const size_t width, const size_t height,
    const size_t numOfPixels,
    const greyScaleArgs *dargs)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = width * y + x;

    if (i < numOfPixels)
    {
        dstimg[i] = pow(255, 1 - dargs->gamma) * pow(dstimg[i], dargs->gamma);
    }
}

void greyScaleTransform(
    const cv::Mat &image,
    const greyScaleArgs &args,
    void (*greyScaleTransformFunc)(unsigned char *, const size_t, const size_t, const size_t, const greyScaleArgs *),
    const int blockWidth,
    const int blockHeight)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int width = image.cols;
    const int height = image.rows;

    greyScaleArgs *dargs;
    CUDADBG(cudaMalloc(&dargs, sizeof(greyScaleArgs) + numOfPixels * sizeof(unsigned char)), );
    unsigned char *dstimg = (unsigned char *)(dargs + 1);

    CUDADBG(cudaMemcpy(dstimg, image.data, numOfPixels * sizeof(unsigned char), cudaMemcpyHostToDevice), );
    CUDADBG(cudaMemcpy(dargs, &args, sizeof(greyScaleArgs), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((width - 1) / blockWidth + 1, (height - 1) / blockHeight + 1, 1);

    greyScaleTransformFunc<<<gridSize, blockSize>>>(dstimg, width, height, numOfPixels, dargs);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost), );
    dstimg = NULL;
    CUDADBG(cudaFree(dargs), );
}