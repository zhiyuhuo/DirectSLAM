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
    CameraIntrinsic* mK;
    TrackingState     mState;

    Frame mRefFrame;
    std::vector<Frame> mFrameVecBuffer;

public: // functions
    bool SetRefFrame(Frame& f);
    bool AddObvFrame(Frame& f);
    TrackingState TrackMono(cv::Mat image, std::vector<float> R_, std::vector<float> t_);

    bool BatchOptimizeSE3Depth(int levelNum);
    
    void CheckReprojection(Frame& ref, Frame& f);
    void ShowDepth(Frame& f);
};

#endif