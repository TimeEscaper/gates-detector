#ifndef GATES_LOCATOR_IMGPROCPIPELINE_H
#define GATES_LOCATOR_IMGPROCPIPELINE_H

#define PIPELINE_DEBUG

#include <functional>
#include <opencv2/opencv.hpp>

class ImgprocPipe {

private:

    cv::Mat currentImage;

public:

    ImgprocPipe(const cv::Mat& currentImage);
    ImgprocPipe(ImgprocPipe& other);
    ~ImgprocPipe();

    ImgprocPipe& operator=(const ImgprocPipe& other);

    ImgprocPipe apply(std::function<cv::Mat(const cv::Mat&)> imgprocFuntion, const char* name = "Operation");
    ImgprocPipe apply(std::function<void(const cv::Mat&, cv::Mat&)> imgprocFuntion, const char* name = "Operation");
    cv::Mat getImage();

};

static ImgprocPipe createPipeline(const cv::Mat src) {
    ImgprocPipe pipe(src);
    return pipe;
}

#endif //GATES_LOCATOR_IMGPROCPIPELINE_H
