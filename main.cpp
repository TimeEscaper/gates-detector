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
    //show("Blue anf green", bgChannel);

    cv::Mat hsv;
    cv::cvtColor(bgChannel, hsv, CV_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    cv::Mat hChannel = hsvChannels[0];
    //show("Hue", hChannel);

    //customLinearTransform(hChannel);
    //cv::blur(hChannel, hChannel, cv::Size(7,7));
    //show("Transformed Hue", hChannel);
    return hChannel;

}

void closing(cv::Mat &image) {
    cv::Mat structElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(2, 2));
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, structElement);
}

int main() {

    cv::Mat src = cv::imread(EXAMPLE_IMAGE);
    //cv::medianBlur(src, src, 15);
    show("Source", src);

    cv::Mat hChannel = hueRescale(src);
    closing(hChannel);
    show("Hue rescaled", hChannel);


    cv::fastNlMeansDenoising(hChannel, hChannel);
    show("NL denoising", hChannel);

    cv::Mat binary;
    cv::adaptiveThreshold(hChannel, binary, MAX_INTENSITY,
                          CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 1);
    show("Adaptive threshold", binary);

    /**
    closing(binary);
    show("Closing", binary);
     */

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    cv::drawContours(src, contours, -1, cv::Scalar(255, 0, 0), 1, 8, hierarchy);
    show("Contours", binary);

    cv::Mat edges;
    cv::Canny(binary, edges, 50, 150);
    show("Canny", edges);

    /**
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/360, 80);
    for (int i = 0; i < lines.size(); i++) {
        cv::line(src, cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0,0,255), 1, 8);
    }
    show("Lines", src);
     */

    return 0;
}