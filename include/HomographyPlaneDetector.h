#ifndef DIRECTSLAM_HPD_H
#define DIRECTSLAM_HPD_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Log.h"
#include "Timer.h"
#include "CameraIntrinsic.h"
#include "Frame.h"
#include "Optimization.h"

enum PlaneDetectionState {
    VOID,
    INITIALIZING,
    TRACKING,
    END
};

class HomographyPlaneDetector {
public: // constructors and destructors
    HomographyPlaneDetector() = default;
    ~HomographyPlaneDetector() = default;
    HomographyPlaneDetector(CameraIntrinsic* K);

public: // members
    int mLevel;
    int mFrameNum;
    CameraIntrinsic* mK;
    PlaneDetectionState mState;

    Frame mRefFrame;
    std::vector<Frame> mFrameVecBuffer;

public: // functions
    bool   SetRefFrame(Frame& f);
    bool   AddObvFrame(Frame& f);
    PlaneDetectionState TrackMono(cv::Mat image, std::vector<float> R_, std::vector<float> t_);

    void   DetectMatchByOpticalFlow(Frame& ref, Frame& f);
    void   ComputeHomographyFromMatchedPoints(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, std::vector<int>& indexsPlane);
    std::vector<float> CheckHomographyReprojError(cv::Mat H, std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1);
    float  GetPatchIntense(float u, float v, int width, unsigned char* image);
};

#endif