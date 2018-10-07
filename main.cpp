#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "include/ImgprocLogicUtil.h"
#include "include/ImgprocPipeline.h"
#include "include/GatesDetector.h"


#define EXAMPLE_IMAGE "/home/sibirsky/gates_locator_images/sauvc-1.png"


int main() {

    cv::Mat src = cv::imread(EXAMPLE_IMAGE);
    show("Source", src);

    cv::Mat image = createPipeline(src, true)
            //.apply(blur, "Blur")
            .apply(gpuMeanShift, "MeanShift")
            .apply(extractValueChannel, "Value channel")
            //.apply(extractLinesSolid, "Draw solid lines")
            //.apply(extractVerticalLines, "Draw vertical lines")
            .apply(morphology, "Closing")
            .getImage();


    GatesDetector gatesDetector;
    GatesDescriptor gates = gatesDetector.detect(image, false);

    if (!gates.hasGates())
        return 0;

    cv::circle(src, gates.getCorners()[0], 10, cv::Scalar(255, 0, 0), 3);
    cv::circle(src, gates.getCorners()[1], 10, cv::Scalar(0, 0, 255), 3);
    cv::circle(src, gates.getCorners()[2], 10, cv::Scalar(255, 0, 0), 3);
    cv::circle(src, gates.getCorners()[3], 10, cv::Scalar(0, 0, 255), 3);
    cv::circle(src, gates.getCenter(), 10, cv::Scalar(0, 255, 0), 3);
    show("Gates", src);

    return 0;
}