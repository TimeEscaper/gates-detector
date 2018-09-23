#include "../include/ImgprocPipeline.h"
#include "../include/ImgprocLogic.h"
#include "../include/Utils.h"

ImgprocPipe::ImgprocPipe(const cv::Mat& currentImage) {
    this->currentImage = currentImage;
}

ImgprocPipe::ImgprocPipe(ImgprocPipe &other) {
    this->currentImage = other.currentImage;
}

ImgprocPipe::~ImgprocPipe() { }

ImgprocPipe& ImgprocPipe::operator=(const ImgprocPipe &other) {
    if (this != &other) {
        this->currentImage = other.currentImage;
    }
    return *this;
}

ImgprocPipe ImgprocPipe::apply(std::function<cv::Mat(const cv::Mat &)> imgprocFuntion, const char* name) {
    cv::Mat newImage = imgprocFuntion(currentImage);
#ifdef PIPELINE_DEBUG
    show(name, newImage);
#endif
    ImgprocPipe pipe(newImage);
    return pipe;
}

ImgprocPipe ImgprocPipe::apply(std::function<void(const cv::Mat &, cv::Mat &)> imgprocFuntion, const char* name) {
    cv::Mat newImage;
    imgprocFuntion(currentImage, newImage);
#ifdef PIPELINE_DEBUG
    show(name, newImage);
#endif
    ImgprocPipe pipe(newImage);
    return pipe;
}

cv::Mat ImgprocPipe::getImage() {
    return currentImage;
}