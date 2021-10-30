// upsampling

#include "../include/sampling.h"

__global__ void bilinear(unsigned char *srcimg, unsigned char *dstimg, int cn, size_t width, size_t newwidth, unsigned factorx, unsigned factory)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = newwidth * y + x;
    unsigned icn = i * cn;

    unsigned distleft = x % factorx;
    unsigned distright = factorx - distleft;
    unsigned disttop = y % factory;
    unsigned distbottom = factory - disttop;

    unsigned nnx = x / factorx;
    unsigned nnx1 = nnx + 1;
    unsigned nny = y / factory;
    unsigned nny1 = nny + 1;

    unsigned nni00cn = (width * nny + nnx) * cn;
    unsigned nni01cn = (width * nny1 + nnx) * cn;
    unsigned nni10cn = (width * nny + nnx1) * cn;
    unsigned nni11cn = (width * nny1 + nnx1) * cn;

    for (unsigned j = 0; j < cn; ++j, ++icn, ++nni00cn, ++nni01cn, ++nni10cn, ++nni11cn)
    {

        dstimg[icn] =
            (srcimg[nni00cn] * distright * distbottom +
             srcimg[nni01cn] * distright * disttop +
             srcimg[nni10cn] * distleft * distbottom +
             srcimg[nni11cn] * distleft * disttop) /
            (factorx * factory);
    }
}
__global__ void NN(unsigned char *srcimg, unsigned char *dstimg, int cn, size_t width, size_t newwidth, unsigned factorx, unsigned factory)
{
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned i = newwidth * y + x;
    unsigned icn = i * cn;

    unsigned distleft = x % factorx;
    unsigned distright = factorx - distleft;

    unsigned nnx = x / factorx;

    if (distleft > distright)
        ++nnx;

    unsigned disttop = y % factory;
    unsigned distbottom = factory - disttop;

    unsigned nny = y / factory;

    if (disttop > distbottom)
        ++nny;

    unsigned nni = width * nny + nnx;
    unsigned nnicn = nni * cn;

    for (unsigned j = 0; j < cn; ++j, ++nnicn, ++icn)
    {
        dstimg[icn] = srcimg[nnicn];
    }
}

void interpolate(cv::Mat &image,
                 void (*interpolateFunc)(unsigned char *srcimg, unsigned char *dstimg, int cn, size_t width, size_t newwidth, unsigned factorx, unsigned factory),
                 const int blockWidth,
                 const int blockHeight,
                 unsigned factorx,
                 unsigned factory)
{
    INITCUDADBG();

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;
    const int newwidth = width * factorx;
    const int newheight = height * factory;

    unsigned char *srcimg;
    CUDADBG(cudaMalloc(&srcimg, (factorx * factory + 1) * numOfElems * sizeof(unsigned char)), );
    unsigned char *dstimg = srcimg + numOfElems;

    CUDADBG(cudaMemcpy(srcimg, image.data, numOfElems * sizeof(unsigned char), cudaMemcpyHostToDevice), );

    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize((newwidth - 1) / blockWidth + 1, (newheight - 1) / blockHeight + 1, 1);

    interpolateFunc<<<gridSize, blockSize>>>(srcimg, dstimg, cn, width, newwidth, factorx, factory);
    CUDACHECK();

    image = cv::Mat(newheight, newwidth, CV_8UC(cn));

    CUDADBG(cudaMemcpy(image.data, dstimg, factorx * factory * numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    dstimg = NULL;
    CUDADBG(cudaFree(srcimg), );
}
