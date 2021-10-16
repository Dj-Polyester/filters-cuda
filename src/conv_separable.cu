#include "../include/conv_separable.h"

__global__ void mooreFilterSeparableH(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const winType *window,
    const unsigned windowRadius,
    const unsigned windowElems,
    const int cn)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
             y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = width * y + x;
    const size_t indexcn = index * cn;

    if (x < width - windowRadius && x > windowRadius - 1)
    {
        winType *sums = new winType[cn];
        winType *windowTotal = new winType[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = windowTotal[i] = 0;

        for (size_t icn = 0, tmp = (index - windowRadius) * cn; icn < windowElems; icn += cn, tmp += cn)
        {
            for (unsigned char j = 0; j < cn; ++j)
            {
                winType windowicnj = window[icn + j];
                sums[j] += windowicnj * srcimg[tmp + j];
                windowTotal[j] += windowicnj;
            }
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
__global__ void mooreFilterSeparableV(
    const unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    const winType *window,
    const unsigned windowRadius,
    const unsigned windowElems,
    const int cn)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
             y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = width * y + x;
    const size_t indexcn = index * cn;

    if (y < height - windowRadius && y > windowRadius - 1)
    {
        winType *sums = new winType[cn];
        winType *windowTotal = new winType[cn];
        for (size_t i = 0; i < cn; ++i)
            sums[i] = windowTotal[i] = 0;
        size_t wcn = width * cn;
        for (size_t icn = 0, tmp = (index - windowRadius * width) * cn; icn < windowElems; icn += cn, tmp += wcn)
        {
            for (unsigned char j = 0; j < cn; ++j)
            {
                winType windowicnj = window[icn + j];
                sums[j] += windowicnj * srcimg[tmp + j];
                windowTotal[j] += windowicnj;
            }
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

void convolveSeparable(
    cv::Mat &image,
    const SeparableWindow &window,
    void (*convolveFuncH)(const unsigned char *, unsigned char *, const size_t, const winType *, const unsigned, const unsigned, const int),
    void (*convolveFuncV)(const unsigned char *, unsigned char *, const size_t, const size_t, const winType *, const unsigned, const unsigned, const int),
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

    convolveFuncH<<<gridSize, blockSize>>>(srcimg, interimg, width, dwindowh, window.radius, window.elems, cn);
    CUDACHECK();
    convolveFuncV<<<gridSize, blockSize>>>(interimg, dstimg, width, height, dwindowv, window.radius, window.elems, cn);
    CUDACHECK();

    CUDADBG(cudaMemcpy(image.data, dstimg, numOfElems * sizeof(unsigned char), cudaMemcpyDeviceToHost), );

    srcimg = dstimg = interimg = NULL;
    dwindowv = NULL;
    CUDADBG(cudaFree(dwindowh), );

    image = cv::Mat(image, cv::Rect(window.radius, window.radius, image.cols - 2 * window.radius, image.rows - 2 * window.radius));
}