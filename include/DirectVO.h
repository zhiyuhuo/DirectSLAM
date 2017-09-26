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

public: // functions
    bool SetFirstFrame(Frame& f);
    TrackingState TrackMono(cv::Mat image);
    bool TrackRefFrame(Frame& f);
};

#endif