
#include "../include/gamma.h"

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

    const size_t numOfPixels = image.total();
    const int cn = image.channels();
    const size_t numOfElems = cn * numOfPixels;
    const int width = image.cols;
    const int height = image.rows;
    std::string filterName(argv[3]);

    if (argc == 5)
    {
        if (filterName == "gammaKernel")
        {
            gammaFilter(image.data, cn, numOfPixels, numOfElems, .600, .200, .200, gammaKernel, std::stoi(argv[4]));
        }
        else if (filterName == "gammaAvgKernel")
        {
            gammaFilter(image.data, cn, numOfPixels, numOfElems, .600, .200, .200, gammaAvgKernel, std::stoi(argv[4]));
        }
        else if (filterName == "greyScale")
        {
            greyScale(image.data, cn, numOfPixels, numOfElems, std::stoi(argv[4]));
        }
    }
    if (argc == 6)
    {
        if (filterName == "gammaKernel2d")
        {
            gammaFilter2d(image.data, cn, width, height, numOfPixels, numOfElems, .600, .200, .200, gammaKernel2d, std::stoi(argv[4]), std::stoi(argv[5]));
        }
        else if (filterName == "gammaAvgKernel2d")
        {
            gammaFilter2d(image.data, cn, width, height, numOfPixels, numOfElems, .600, .200, .200, gammaAvgKernel2d, std::stoi(argv[4]), std::stoi(argv[5]));
        }
        else if (filterName == "greyScale2d")
        {
            greyScale2d(image.data, cn, width, height, numOfPixels, numOfElems, gammaKernel2d, std::stoi(argv[4]), std::stoi(argv[5]));
        }
    }

    cv::imwrite(argv[2], image);
    return 0;
}