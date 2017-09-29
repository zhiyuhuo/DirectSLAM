#ifndef DIRECTSLAM_VO_H
#define DIRECTSLAM_VO_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Log.h"
#include "Timer.h"
#include "CameraIntrinsic.h"
#include "Frame.h"
#include "Optimization.h"

enum TrackingState {
    VOID,
    INITIALIZING,
    TRACKING,
    LOST
};

class DirectVO {
public: // constructors and destructors
    DirectVO() = default;
    ~DirectVO() = default;
    DirectVO(CameraIntrinsic* K);

public: // members
    int mLevel;
    int mFrameNum;
    CameraIntrinsic* mK;
    TrackingState     mState;

    Frame mRefFrame;
    std::vector<Frame> mFrameVecBuffer;

public: // functions
    bool SetRefFrame(Frame& f);
    bool AddObvFrame(Frame& f);
    TrackingState TrackMono(cv::Mat image, std::vector<float> R_, std::vector<float> t_);

    bool BatchOptimizeSE3Depth();
    
    float GetReprojectionPhotometricError(Frame& ref, Frame& f, cv::Point2f p_ref, float d_ref);
    void  CheckErrorForDistances();
    void CheckReprojection(Frame& ref, Frame& f);
    void ShowDepth(Frame& f);
    double GetPatchIntense(float u, float v, int width, unsigned char* image);
};

#endif