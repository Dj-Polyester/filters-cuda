
#include "../include/gamma.h"
#include "../include/convolution.h"
#include "../include/sampling.h"

#define GETWINDOW

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6 && argc != 7 && argc != 8)
        ERROR("usage: executable in out filter dims [windowargs]", );
    // set to page-locked (pinned memory)
    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    cv::Mat image;

    image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (!image.data)
        ERROR("Image null", );

    if (image.empty())
        ERROR("Image empty", );

    if (!image.isContinuous())
        ERROR("image is not continuous", );

    std::string filterName(argv[3]);
    INITCUDADBG();
    INITCUDABENCH();

    if (filterName == "gammaKernel")
    {
        CUDABENCHEXPR(gammaFilter(image, {.200, .600, .200}, gammaKernel, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "gammaAvgKernel")
    {
        CUDABENCHEXPR(gammaFilter(image, {.600, .100, .100}, gammaAvgKernel, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "greyScale")
    {
        CUDABENCHEXPR(greyScale(image, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "mean")
    {
        Window window(std::stoi(argv[6]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolve(image, window, mooreFilter, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "gaussian")
    {
        Window window(std::stoi(argv[6]), image.channels(), WindowType::gaussian, std::stoi(argv[7]));
        CUDABENCHEXPR(convolve(image, window, mooreFilter, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "meanSeparable")
    {
        SeparableWindow window(std::stoi(argv[6]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolveSeparable(image, window, mooreFilterSeparableH, mooreFilterSeparableV, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "gaussianSeparable")
    {
        SeparableWindow window(std::stoi(argv[6]), image.channels(), WindowType::gaussian, std::stoi(argv[7]));
        CUDABENCHEXPR(convolveSeparable(image, window, mooreFilterSeparableH, mooreFilterSeparableV, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "meanShared")
    {
        Window window(std::stoi(argv[6]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolveShared(image, window, mooreFilterShared, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "gaussianShared")
    {
        Window window(std::stoi(argv[6]), image.channels(), WindowType::gaussian, std::stoi(argv[7]));
        CUDABENCHEXPR(convolveShared(image, window, mooreFilterShared, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "meanSharedSep")
    {
        SeparableWindow window(std::stoi(argv[6]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolveSharedSep(image, window, mooreFilterSharedSep, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "gaussianSharedSep")
    {
        SeparableWindow window(std::stoi(argv[6]), image.channels(), WindowType::gaussian, std::stoi(argv[7]));
        CUDABENCHEXPR(convolveSharedSep(image, window, mooreFilterSharedSep, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "NN")
    {
        CUDABENCHEXPR(interpolate(image, NN, std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7])));
    }
    else if (filterName == "bilinear")
    {
        CUDABENCHEXPR(interpolate(image, bilinear, std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7])));
    }

    PRINTCUDABENCH2(MS);
    cv::imwrite(argv[2], image);
    CLEANUPCUDABENCH();
    return 0;
}