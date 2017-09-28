#ifndef DIRECTSLAM_FRAME_H
#define DIRECTSLAM_FRAME_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Log.h"

class Frame {
public:
    Frame() = default;
    ~Frame() = default;
    Frame(const cv::Mat& image, double timeStamp = 0);
    Frame(const Frame& frame);
    void operator=(const Frame& frame);

public:
    std::vector<cv::Mat> mImgPyr;
    std::vector<std::vector<cv::KeyPoint> > mKpsPyr;
    std::vector<std::vector<float> > mDepthPyr;
    float mScaleFactor;
    double mTimeStamp;
    cv::Mat mR;
    cv::Mat mt;

public:
    void ExtractFastPyr();
    void ExtractSlopePyr(int threshold);
    void InitDepthPyr(float initDepth);

    cv::Mat GetDoubleSE3();

public: // for debug and display
    void ShowPyr(int levelShow);

};

class Measure2D {
    cv::Point2f mKeypoint;
    float       mD;
    
};

#endif