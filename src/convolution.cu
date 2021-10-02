// __global__ void mooreFilter(
//     unsigned char *srcimg,
//     unsigned char *dstimg,
//     unsigned char *window,
//     unsigned n, //n should be odd
//     const int cn)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned n2 = n * n, ndiv2 = n / 2;
//     if (x < width - ndiv2 && x > ndiv2 - 1 && y < height - ndiv2 && y > ndiv2 - 1)
//     {
//         unsigned sum = 0;
//         for (unsigned i = 0, tmp = index - n2 / 2; i < n2; ++i, ++tmp)
//             sum += window[i] * srcimg[tmp];

//         dstimg[index] = sum;
//     }
//     else
//     {
//         dstimg[index] = srcimg[index];
//     }
// }
__global__ void mooreFilter2d(
    unsigned char *srcimg,
    unsigned char *dstimg,
    const size_t width,
    const size_t height,
    unsigned char *window,
    unsigned n, //n should be odd
    const int cn)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = width * y + x;
    unsigned n2 = n * n, ndiv2 = n / 2;
    if (x < width - ndiv2 && x > ndiv2 - 1 && y < height - ndiv2 && y > ndiv2 - 1)
    {
        unsigned sum = 0;
        for (unsigned i = 0, tmp = index - n2 / 2; i < n2; ++i, ++tmp)
            sum += window[i] * srcimg[tmp];

        dstimg[index] = sum;
    }
    else
    {
        dstimg[index] = srcimg[index];
    }
}