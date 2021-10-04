
#include "../include/gamma.h"
#include "../include/convolution.h"

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6)
        ERROR("usage: executable in out filter args", );

    cv::Mat image;

    image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (!image.data)
        ERROR("Image null", );

    if (image.empty())
        ERROR("Image empty", );

    if (!image.isContinuous())
        ERROR("image is not continuous", );

    std::string filterName(argv[3]);
    INITCUDABENCH();
    if (argc == 5)
    {
        if (filterName == "gammaKernel")
            CUDABENCH(gammaFilter(image, {.600, .200, .200}, gammaKernel, std::stoi(argv[4])));

        else if (filterName == "gammaAvgKernel")
            CUDABENCH(gammaFilter(image, {.600, .200, .200}, gammaAvgKernel, std::stoi(argv[4])));

        else if (filterName == "greyScale")
            CUDABENCH(greyScale(image, std::stoi(argv[4])));
    }
    if (argc == 6)
    {
        if (filterName == "gammaKernel2d")
            CUDABENCH(gammaFilter2d(image, {.200, .600, .200}, gammaKernel2d, std::stoi(argv[4]), std::stoi(argv[5])));

        else if (filterName == "gammaAvgKernel2d")
            CUDABENCH(gammaFilter2d(image, {.600, .100, .100}, gammaAvgKernel2d, std::stoi(argv[4]), std::stoi(argv[5])));

        else if (filterName == "greyScale2d")
            CUDABENCH(greyScale2d(image, std::stoi(argv[4]), std::stoi(argv[5])));

        else if (filterName == "convolve2d")
        {
            Window window(3, image.channels(), WindowType::mean);
            convolve2d(image, window, mooreFilter2d, std::stoi(argv[4]), std::stoi(argv[5]));
        }
    }
    PRINTCUDABENCH2(MS)
    cv::imwrite(argv[2], image);
    return 0;
}