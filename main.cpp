#include <iostream>

#define IMAGES_DIR "/home/sibirsky/gates_locator_images/"
#define EXAMPLE_IMAGE "/home/sibirsky/gates_locator_images/gates.jpg"
#define MAX_INTENSITY 360

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

typedef uchar Pixel;

void show(const char* name, const cv::Mat &image) {
    cv::namedWindow(name);
    cv::imshow(name, image);
    cv::waitKey();
    cv::destroyWindow(name);
}

void customLinearTransform(cv::Mat &image) {
    double max, min;
    cv::minMaxLoc(image, &min, &max);
    double diff = max - min;
    image.forEach<Pixel>(
            [min, diff](Pixel &pixel, const int *position) -> void {
                pixel = ((pixel - min) / diff) * MAX_INTENSITY;
            });
}

cv::Mat hueRescale(cv::Mat image) {
    cv::Mat rgbChannels[3];
    cv::split(image, rgbChannels);
    rgbChannels[2] = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::Mat bgChannel;
    cv::merge(rgbChannels, 3, bgChannel);

    cv::Mat hsv;
    cv::cvtColor(bgChannel, hsv, CV_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    cv::Mat hChannel = hsvChannels[0];

    customLinearTransform(hChannel);

    return hChannel;
}

void closing(cv::Mat &image) {
    cv::Mat structElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(2, 2));
    for (int i = 1; i <= 1; i++) {
        cv::morphologyEx(image, image, cv::MORPH_CLOSE, structElement, cv::Point(-1,-1), i);
    }
}

int main() {

    cv::Mat src = cv::imread(EXAMPLE_IMAGE);
    //cv::medianBlur(src, src, 15);
    show("Source", src);

    cv::Mat hue = hueRescale(src);
    show("Hue", hue);

    cv::Mat denoised;
    cv::fastNlMeansDenoising(hue, denoised);
    show("NL denoise", denoised);

    cv::Mat bw;
    cv::adaptiveThreshold(denoised, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 7, 1);
    closing(bw);
    show("Adaptive threshold", bw);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    cv::drawContours(src, contours, -1, cv::Scalar(0, 0, 255), 1, 8, hierarchy);

    std::vector<std::vector<cv::Point>> hulls(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        cv::convexHull(contours[i], hulls[i]);
    }
    cv::drawContours(src, hulls, -1, cv::Scalar(255, 0, 0), 1, 8, hierarchy);
    show("Hulls", src);

    /**TODO: find max area hull and get its center */

    return 0;
}