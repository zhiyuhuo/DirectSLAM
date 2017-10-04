#ifndef __ANDROID__

#ifndef DIRECTSLAM_VIEWER_H
#define DIRECTSLAM_VIEWER_H

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <pangolin/pangolin.h>

#include <mutex>

#include "PlaneDetector.h"

class ImageFrame;
class CameraIntrinsic;

class Viewer
{
    public:
        Viewer() = default;
        Viewer(PlaneDetector* _pd): pd(_pd), isFinished(false) {};

        // 3D Drawing
        void run();
        void Stop();

        void drawRefFrame();
        void drawFrames();
        void drawPixelsDepth();
        void drawMapAxis();


    private:
        PlaneDetector* pd;

        std::mutex mMutexFinish;
        bool isFinished;

        pangolin::OpenGlRenderState* s_cam;
        pangolin::View* d_cam;
        pangolin::OpenGlMatrix M;

};

#endif   // VIEWER_H

#endif  // __ANDROID__
