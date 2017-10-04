#ifndef __ANDROID__

#include "Viewer.h"

void Viewer::run()
{
    pangolin::CreateWindowAndBind("Viewer", 800, 600);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = new pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 
            512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -2.5, -5, 0, 0, 0, 0, -1, 0)
            );

    d_cam = & ( pangolin::CreateDisplay()
        .SetBounds(0, 1, pangolin::Attach::Pix(175), 1.f, -1024.f/768.f)
        .SetHandler(new pangolin::Handler3D(*s_cam)) );

    M.SetIdentity();

    bool bFollow = true;

    while(1) {

        {
            std::unique_lock<std::mutex> lock(mMutexFinish);
            if (isFinished) {
                isFinished = false;
                break;
            }
        }

        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam->Activate(*s_cam);
        glClearColor(1.f, 1.f, 1.f, 1.f);

        drawRefFrame();

        drawFrames();

        drawPixelsDepth();

        drawMapAxis();

        pangolin::FinishFrame();

        usleep(30000);
    }

}

void Viewer::drawRefFrame()
{
    const float w = 1.0;
    const float h = w*0.75;
    const float z = w*0.6;

    if (pd == NULL)
        return;

    if (!pd->mRefFrame.mR.empty()) {
        cv::Mat Tcw = pd->mRefFrame.GetTcwMat().t();

        glPushMatrix();

        glMultMatrixf(Tcw.ptr<float>(0));

        glLineWidth(2);

        glColor3f(0.0f,1.0f,1.0f);

        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }

}

void Viewer::drawFrames()
{
    const float w = 0.5;
    const float h = w*0.75;
    const float z = w*0.6;

    if (pd==NULL)
        return;

    for (int i = 0; i < pd->mFrameVecBuffer.size(); i++) {

        cv::Mat Tcw = pd->mFrameVecBuffer[i].GetTcwMat().t();

        glPushMatrix();

        glMultMatrixf(Tcw.ptr<float>(0));

        glLineWidth(2);

        glColor3f(0.0f,0.0f,1.0f);

        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();

    }

}

void Viewer::drawPixelsDepth()
{
    if (pd==NULL || pd->mRefFrame.mKpsPyr.size() <= pd->mLevel)
        return;

    int Level = pd->mLevel;
    glPointSize(3);
    glBegin(GL_POINTS);

    float fx = pd->mK->fx / pow(pd->mRefFrame.mScaleFactor, Level);
    float fy = pd->mK->fy / pow(pd->mRefFrame.mScaleFactor, Level);
    float cx = pd->mK->cx / pow(pd->mRefFrame.mScaleFactor, Level);
    float cy = pd->mK->cy / pow(pd->mRefFrame.mScaleFactor, Level);

    // local map
    float x, y, z;
    for (int i = 0; i < pd->mRefFrame.mKpsPyr[Level].size(); i++) {
        
        z =  pd->mRefFrame.mDepthPyr[Level][i];
        x = (pd->mRefFrame.mKpsPyr[Level][i].pt.x - cx) * z / fx;
        y = (pd->mRefFrame.mKpsPyr[Level][i].pt.y - cy) * z / fy;

        glColor3f(0, 0.f, 0);
        glVertex3f(x, y, z);

        if (pd->mRefFrame.mStatisticPyr[Level][i].getNum() == pd->mFrameNum 
            &&  pd->mRefFrame.mStatisticPyr[Level][i].getMean() < 5.99
            &&  pd->mRefFrame.mStatisticPyr[Level][i].getStdVariance() < 1.99) {
            // glColor3f(0, 1.f, 0);
            // glVertex3f(x, y, z);
        }
        else if (pd->mRefFrame.mStatisticPyr[Level][i].getNum() > pd->mFrameNum / 2
        &&  pd->mRefFrame.mStatisticPyr[Level][i].getMean() < 4.99
        &&  pd->mRefFrame.mStatisticPyr[Level][i].getStdVariance() < 2.99) {
            // glColor3f(.5f, .5f, 0);
            // glVertex3f(x, y, z);
        }
        else if (pd->mRefFrame.mStatisticPyr[Level][i].getNum() > pd->mFrameNum / 10) {
            // glColor3f(1.f, 0, 0);
            // glVertex3f(x, y, z);
        }
    }

    glEnd();

}

void Viewer::drawMapAxis()
{   
    glLineWidth(2);
    
    // Z Axis blue
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(0,0,1);
    glEnd();

    // Y Axis red
    glColor3f(1.0f,0.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(0,1,0);
    glEnd();

    // X Axis green
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(1,0,0);
    glEnd();
}


void Viewer::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    isFinished = true;
}

#endif    // __ANDROID__
