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

    for (int i = 0; i < pd->mFramesBuffer.size(); i++) {

        cv::Mat Tcw = pd->mFramesBuffer[i].GetTcwMat().t();

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

void Viewer::DrawPlane(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane, std::vector<cv::Point3f> points)
{
    cv::Mat res;
    cv::cvtColor(f.mImg, res, CV_GRAY2BGR);

    float a = mainPlane[0];
    float b = mainPlane[1];
    float c = mainPlane[2];
    float d = mainPlane[3];

    cv::Point3f sZ(a, b, c);
    float lz = sqrt(a*a + b*b + c*c);
    sZ = sZ / lz;

    cv::Point3f sX(-d/a, 0, d/c);
    float lx = sqrt(sX.x*sX.x + sX.y*sX.y + sX.z*sX.z);
    sX = sX / lx;

    cv::Point3f sY(sZ.y*sX.z-sZ.z*sX.y, sZ.z*sX.x-sZ.x*sX.z, sZ.x*sX.y-sZ.y*sX.x);
    float ly = sqrt(sY.x*sY.x + sY.y*sY.y + sY.z*sY.z);
    sY = sY / ly;

    cv::Point3f Op(0, 0, -d/c);
    float edgeLengh = 0.075;
    float eL = edgeLengh;

    cv::Point3f LT, RT, LB, RB;
    cv::Point2f LT2, RT2, LB2, RB2;
    for (int i = 0; i < points.size(); i++) {
        // float cx = ?;
        // float cy = ?;
        // O = Op + sX*cx + sY*cy;
        cv::Point3f O = points[i];
        // draw four corners
        cv::Point3f LT = O + eL * (sY)  + eL * (-sX);
        cv::Point3f RT = O + eL * (sY)  + eL * (sX);
        cv::Point3f LB = O + eL * (-sY) + eL * (-sX);
        cv::Point3f RB = O + eL * (-sY) + eL * (sX);

        GetAxis(K, f, LT, RT, LT2, RT2);
        if (LT2.x > 0 && LT2.x < res.cols && RT2.y > 0 && RT2.y < res.rows)
            cv::line(res, LT2, RT2, cv::Scalar(0, 255, 255), 1);
    
        GetAxis(K, f, LB, RB, LB2, RB2);
        if (LB2.x > 0 && LB2.x < res.cols && RB2.y > 0 && RB2.y < res.rows)
            cv::line(res, LB2, RB2, cv::Scalar(0, 255, 255), 1);
    
        if (LT2.x > 0 && LT2.x < res.cols && LB2.y > 0 && LB2.y < res.rows)
            cv::line(res, LT2, LB2, cv::Scalar(0, 255, 255), 1);
    
        if (RT2.x > 0 && RT2.x < res.cols && RB2.y > 0 && RB2.y < res.rows)
            cv::line(res, RT2, RB2, cv::Scalar(0, 255, 255), 1);

    }

    cv::imshow("Plane", res);

}

void Viewer::DrawSquare(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane, float sqaureCentroidX, float sqaureCentroidY)
{
    cv::Mat res;
    cv::cvtColor(f.mImg, res, CV_GRAY2BGR);

    float a = mainPlane[0];
    float b = mainPlane[1];
    float c = mainPlane[2];
    float d = mainPlane[3];

    float cx = sqaureCentroidX;
    float cy = sqaureCentroidY;

    cv::Point3f sZ(a, b, c);
    float lz = sqrt(a*a + b*b + c*c);
    sZ = sZ / lz;

    //cv::Point3f sX(1, 0, (-d-a)/c - (-d/c));
    cv::Point3f sX(-d/a, 0, d/c);
    float lx = sqrt(sX.x*sX.x + sX.y*sX.y + sX.z*sX.z);
    sX = sX / lx;

    cv::Point3f sY(sZ.y*sX.z-sZ.z*sX.y, sZ.z*sX.x-sZ.x*sX.z, sZ.x*sX.y-sZ.y*sX.x);
    float ly = sqrt(sY.x*sY.x + sY.y*sY.y + sY.z*sY.z);
    sY = sY / ly;

    // draw AR Axis
    cv::Point3f Op(0, 0, -d/c);
    float edgeLengh = 0.1;
    float eL = edgeLengh;

    cv::Point3f O = Op + sX*cx + sY*cy;
    // draw four corners
    cv::Point3f LT = O + eL * (sY)  + eL * (-sX);
    cv::Point3f RT = O + eL * (sY)  + eL * (sX);
    cv::Point3f LB = O + eL * (-sY) + eL * (-sX);
    cv::Point3f RB = O + eL * (-sY) + eL * (sX);

    cv::Point2f LT2, RT2;
    GetAxis(K, f, LT, RT, LT2, RT2);
    if (LT2.x > 0 && LT2.x < res.cols && RT2.y > 0 && RT2.y < res.rows)
        cv::line(res, LT2, RT2, cv::Scalar(0, 255, 255), 1);

    cv::Point2f LB2, RB2;
    GetAxis(K, f, LB, RB, LB2, RB2);
    if (LB2.x > 0 && LB2.x < res.cols && RB2.y > 0 && RB2.y < res.rows)
        cv::line(res, LB2, RB2, cv::Scalar(0, 255, 255), 1);

    if (LT2.x > 0 && LT2.x < res.cols && LB2.y > 0 && LB2.y < res.rows)
        cv::line(res, LT2, LB2, cv::Scalar(0, 255, 255), 1);

    if (RT2.x > 0 && RT2.x < res.cols && RB2.y > 0 && RB2.y < res.rows)
        cv::line(res, RT2, RB2, cv::Scalar(0, 255, 255), 1);

    cv::imshow("Square", res);
}


void Viewer::DrawAR(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane)
{
    cv::Mat res;
    cv::cvtColor(f.mImg, res, CV_GRAY2BGR);

    float a = mainPlane[0];
    float b = mainPlane[1];
    float c = mainPlane[2];
    float d = mainPlane[3];


    cv::Point3f sZ(a, b, c);
    float lz = sqrt(a*a + b*b + c*c);
    sZ = sZ / lz;

    //cv::Point3f sX(1, 0, (-d-a)/c - (-d/c));
    cv::Point3f sX(-d/a, 0, d/c);
    float lx = sqrt(sX.x*sX.x + sX.y*sX.y + sX.z*sX.z);
    sX = sX / lx;

    cv::Point3f sY(sZ.y*sX.z-sZ.z*sX.y, sZ.z*sX.x-sZ.x*sX.z, sZ.x*sX.y-sZ.y*sX.x);
    float ly = sqrt(sY.x*sY.x + sY.y*sY.y + sY.z*sY.z);
    sY = sY / ly;

    // draw AR Axis
    cv::Point3f O(0, 0, -d/c);

    cv::Point3f Z = O + sZ;
    cv::Point3f X = O + sX;
    cv::Point3f Y = O + sY;

    cv::Point2f OX, XX;
    GetAxis(K, f, O, X, OX, XX);
    if (OX.x > 0 && OX.x < res.cols && OX.y > 0 && OX.y < res.rows)
        cv::line(res, OX, XX, cv::Scalar(0, 255, 0), 2);

    cv::Point2f OY, YY;
    GetAxis(K, f, O, Y, OY, YY);
    if (OY.x > 0 && OY.x < res.cols && OY.y > 0 && OY.y < res.rows)
        cv::line(res, OY, YY, cv::Scalar(0, 0, 255), 2);

    cv::Point2f OZ, ZZ;
    GetAxis(K, f, O, Z, OZ, ZZ);
    if (OZ.x > 0 && OZ.x < res.cols && OZ.y > 0 && OZ.y < res.rows)
        cv::line(res, OZ, ZZ, cv::Scalar(255, 0, 0), 2);

    cv::imshow("AR", res);
}

void Viewer::DrawAR(CameraIntrinsic* K, Frame& f, std::vector<float> mainPlane, std::vector<float> anchorPoint)
{
    cv::Mat res;
    cv::cvtColor(f.mImg, res, CV_GRAY2BGR);

    float a = mainPlane[0];
    float b = mainPlane[1];
    float c = mainPlane[2];
    float d = mainPlane[3];


    cv::Point3f sZ(a, b, c);
    float lz = sqrt(a*a + b*b + c*c);
    sZ = sZ / lz;

    //cv::Point3f sX(1, 0, (-d-a)/c - (-d/c));
    cv::Point3f sX(-d/a, 0, d/c);
    float lx = sqrt(sX.x*sX.x + sX.y*sX.y + sX.z*sX.z);
    sX = sX / lx;

    cv::Point3f sY(sZ.y*sX.z-sZ.z*sX.y, sZ.z*sX.x-sZ.x*sX.z, sZ.x*sX.y-sZ.y*sX.x);
    float ly = sqrt(sY.x*sY.x + sY.y*sY.y + sY.z*sY.z);
    sY = sY / ly;

    // draw AR Axis
    cv::Point3f O(anchorPoint[0], anchorPoint[1], anchorPoint[2]);

    cv::Point3f Z = O + sZ;
    cv::Point3f X = O + sX;
    cv::Point3f Y = O + sY;

    cv::Point2f OX, XX;
    GetAxis(K, f, O, X, OX, XX);
    if (OX.x > 0 && OX.x < res.cols && OX.y > 0 && OX.y < res.rows)
        cv::line(res, OX, XX, cv::Scalar(0, 255, 0), 2);

    cv::Point2f OY, YY;
    GetAxis(K, f, O, Y, OY, YY);
    if (OY.x > 0 && OY.x < res.cols && OY.y > 0 && OY.y < res.rows)
        cv::line(res, OY, YY, cv::Scalar(0, 0, 255), 2);

    cv::Point2f OZ, ZZ;
    GetAxis(K, f, O, Z, OZ, ZZ);
    if (OZ.x > 0 && OZ.x < res.cols && OZ.y > 0 && OZ.y < res.rows)
        cv::line(res, OZ, ZZ, cv::Scalar(255, 0, 0), 2);

    cv::imshow("AR with Main Plane and Anchor Point", res);
}

void Viewer::GetAxis(CameraIntrinsic* K, Frame& f, const cv::Point3f& p3d1, const cv::Point3f &p3d2,
                cv::Point2f& p2d1, cv::Point2f& p2d2)
{
    float* _R = f.mR.ptr<float>(0);
    float* _t = f.mt.ptr<float>(0);

    float X1 = _R[0]*p3d1.x + _R[1]*p3d1.y + _R[2]*p3d1.z + _t[0];
    float Y1 = _R[3]*p3d1.x + _R[4]*p3d1.y + _R[5]*p3d1.z + _t[1];
    float Z1 = _R[6]*p3d1.x + _R[7]*p3d1.y + _R[8]*p3d1.z + _t[2];
    float X2 = _R[0]*p3d2.x + _R[1]*p3d2.y + _R[2]*p3d2.z + _t[0];
    float Y2 = _R[3]*p3d2.x + _R[4]*p3d2.y + _R[5]*p3d2.z + _t[1];
    float Z2 = _R[6]*p3d2.x + _R[7]*p3d2.y + _R[8]*p3d2.z + _t[2];
    //Log_debug("P1: {} {} {} P2: {} {} {}", X1, Y1, Z1, X2, Y2, Z2);

    float near = 1.f;

    if (Z1 <= near && Z2 <= near) {
        //Log_error("Axis 2 point z < 0, return (");
        p2d1.x = p2d1.y = -999999;
        p2d2.x = p2d2.y = -999999;
        return;
    }

    if (Z1 <= near || Z2 <= near) {
        if (Z1 <= near) {
            float ratio = (near - Z1) / (Z2 - Z1);
            X1 = X1 + (X2 - X1)*ratio;
            Y1 = Y1 + (Y2 - Y1)*ratio;
            Z1 = near;
        } else {
            float ratio = (near - Z2) / (Z1 - Z2);
            X2 = X2 + (X1 - X2)*ratio;
            Y2 = Y2 + (Y1 - Y2)*ratio;
            Z2 = near;
        }
    }

    p2d1 = cv::Point2f(K->fx*X1/Z1+K->cx, K->fy*Y1/Z1+K->cy);
    p2d2 = cv::Point2f(K->fx*X2/Z2+K->cx, K->fy*Y2/Z2+K->cy);
    // Log_debug("Distort P1: {} {} P2: {} {}", p2d1.x, p2d1.y, p2d2.x, p2d2.y);
}

#endif    // __ANDROID__
