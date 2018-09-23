#include <opencv2/highgui.hpp>
#include "../include/Utils.h"

void show(const char* name, const cv::Mat &image) {
    cv::namedWindow(name);
    cv::imshow(name, image);
    cv::waitKey();
    cv::destroyWindow(name);
}