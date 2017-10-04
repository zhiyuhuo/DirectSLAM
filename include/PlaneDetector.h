#ifndef DIRECTSLAM_PD_H
#define DIRECTSLAM_PD_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Log.h"
#include "Timer.h"
#include "CameraIntrinsic.h"
#include "Frame.h"
#include "TextureSegment.h"

enum PlaneDetectionState {
    VOID,
    INITIALIZING,
    TRACKING,
    END
};

class PlaneDetector {
public: // constructors and destructors
    PlaneDetector() = default;
    ~PlaneDetector() = default;
    PlaneDetector(CameraIntrinsic* K);

public: // members
    int mLevel;
    int mFrameNum;
    CameraIntrinsic* mK;
    PlaneDetectionState mState;

    Frame mRefFrame;
    std::vector<Frame> mFrameVecBuffer;

    // function members
    TextureSegment mTextureSeg;

public: // functions
    bool   SetRefFrame(Frame& f);
    bool   AddObvFrame(Frame& f);
    PlaneDetectionState TrackMono(cv::Mat image, std::vector<float> R_, std::vector<float> t_);

    void   DetectMatchByOpticalFlow(Frame& ref, Frame& f);
    cv::Mat   ComputeHomographyFromMatchedPoints(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, std::vector<int>& indexsPlane);
    std::vector<float> CheckHomographyReprojError(cv::Mat H, std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1);
    cv::Mat ExtractPlaneFromHomographyAndRT(cv::Mat H01, cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1);
    float  GetPatchIntense(float u, float v, int width, unsigned char* image);
    
};

#endif