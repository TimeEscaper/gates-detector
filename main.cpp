#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "include/ImgprocLogicUtil.h"
#include "include/ImgprocPipeline.h"
#include "include/GatesDetector.h"


#define EXAMPLE_IMAGE "/home/sibirsky/gates_locator_images/sauvc-1.png"

void testDetector(const cv::Mat& src);

int main() {

    std::vector<std::string> testImages;
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-1.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-2.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-3.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-4.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-5.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-6.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-7.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-8.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-9.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/sauvc-10.png");
    testImages.push_back("/home/sibirsky/gates_locator_images/gates.jpg");
    testImages.push_back("/home/sibirsky/gates_locator_images/frame1-1002.jpg");
    testImages.push_back("/home/sibirsky/gates_locator_images/frame2-1059.jpg");

    for (int i = 0; i < testImages.size(); i++) {
        cv::Mat src = cv::imread(testImages[i]);
        testDetector(src);
    }

    return 0;
}

void testDetector(const cv::Mat& src) {
    cv::Mat canvas = src;
    show("New source", canvas);

    cv::Mat image = createPipeline(src, false)
            //.apply(blur, "Blur")
            .apply(gpuMeanShift, "MeanShift")
            .apply(extractValueChannel, "Value channel")
                    //.apply(extractLinesSolid, "Draw solid lines")
                    //.apply(extractVerticalLines, "Draw vertical lines")
            .apply(morphology, "Closing")
            .getImage();

    GatesDetector detector;
    GatesDescriptor gates = detector.detect(image, false);

    if (!gates.hasGates())
        return;

    cv::circle(src, gates.getCorners()[0], 10, cv::Scalar(255, 0, 0), 3);
    cv::circle(src, gates.getCorners()[1], 10, cv::Scalar(0, 0, 255), 3);
    cv::circle(src, gates.getCorners()[2], 10, cv::Scalar(255, 0, 0), 3);
    cv::circle(src, gates.getCorners()[3], 10, cv::Scalar(0, 0, 255), 3);
    cv::circle(src, gates.getCenter(), 10, cv::Scalar(0, 255, 0), 3);
    show("Gates", canvas);
}