
#include "../include/gamma.h"
#include "../include/convolution.h"

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6 && argc != 7)
        ERROR("usage: executable in out filter args", );
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
        CUDABENCHEXPR(gammaFilter(image, {.600, .200, .200}, gammaKernel, std::stoi(argv[4])));
    }

    else if (filterName == "gammaAvgKernel")
    {
        CUDABENCHEXPR(gammaFilter(image, {.600, .200, .200}, gammaAvgKernel, std::stoi(argv[4])));
    }

    else if (filterName == "greyScale")
    {
        CUDABENCHEXPR(greyScale(image, std::stoi(argv[4])));
    }
    else if (filterName == "convolve")
    {
        Window window(std::stoi(argv[5]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolve(image, window, mooreFilter, std::stoi(argv[4])));
    }
    else if (filterName == "convolveSeparable")
    {
        SeparableWindow window(std::stoi(argv[5]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolveSeparable(image, window, mooreFilterSeparableH, mooreFilterSeparableV, std::stoi(argv[4])));
    }
    else if (filterName == "gammaKernel2d")
    {
        CUDABENCHEXPR(gammaFilter2d(image, {.200, .600, .200}, gammaKernel2d, std::stoi(argv[4]), std::stoi(argv[5])));
    }

    else if (filterName == "gammaAvgKernel2d")
    {
        CUDABENCHEXPR(gammaFilter2d(image, {.600, .100, .100}, gammaAvgKernel2d, std::stoi(argv[4]), std::stoi(argv[5])));
    }

    else if (filterName == "greyScale2d")
    {
        CUDABENCHEXPR(greyScale2d(image, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "convolve2d")
    {
        Window window(std::stoi(argv[6]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolve2d(image, window, mooreFilter2d, std::stoi(argv[4]), std::stoi(argv[5])));
    }
    else if (filterName == "convolve2dSeparable")
    {
        SeparableWindow window(std::stoi(argv[6]), image.channels(), WindowType::mean);
        CUDABENCHEXPR(convolve2dSeparable(image, window, mooreFilter2dSeparableH, mooreFilter2dSeparableV, std::stoi(argv[4]), std::stoi(argv[5])));
    }

    PRINTCUDABENCH2(MS);
    cv::imwrite(argv[2], image);
    CLEANUPCUDABENCH();
    return 0;
}