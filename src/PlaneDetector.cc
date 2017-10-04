#include "PlaneDetector.h" 
#define LEVEL 0
#define FRAMENUM 5

PlaneDetector::PlaneDetector(CameraIntrinsic* K)
{
    mK = K;
    mState = PlaneDetectionState::VOID;
    mLevel = LEVEL;
    mFrameNum = FRAMENUM;
}

bool PlaneDetector::SetRefFrame(Frame& f)
{
    f.ExtractFeaturePoint();
    f.ShowPyr(mLevel);

    mRefFrame = f;
    return true;
}

bool PlaneDetector::AddObvFrame(Frame& f)
{
    mFrameVecBuffer.push_back(f);
    return true;
}

PlaneDetectionState PlaneDetector::TrackMono(cv::Mat image, std::vector<float> R_,  std::vector<float> t_)
{
    Frame f(image);

    if (R_.size() > 0 && t_.size() > 0) {
        std::memcpy(f.mR.ptr<float>(0), R_.data(), R_.size()*sizeof(float));
        std::memcpy(f.mt.ptr<float>(0), t_.data(), t_.size()*sizeof(float));
    }

    if (mState == PlaneDetectionState::VOID) {
        SetRefFrame(f);
        mTextureSeg.InitData(f.mImgPyr[mLevel], 20, 20);
        mTextureSeg.ComputePatchFeatures();
        mState = PlaneDetectionState::INITIALIZING;
    }
    else if (mState == PlaneDetectionState::INITIALIZING) {
        AddObvFrame(f);
        if (mFrameVecBuffer.size() >= mFrameNum) {
            mState = PlaneDetectionState::TRACKING;
        }
    }
    else if (mState == PlaneDetectionState::TRACKING) {
        TIME_BEGIN();
        DetectMatchByOpticalFlow(mRefFrame, mFrameVecBuffer[3]);
        TIME_END("DetectMatchByOpticalFlow");
        mState = PlaneDetectionState::END;
    }
    else {
        ;
    }

    return mState;
}

void PlaneDetector::DetectMatchByOpticalFlow(Frame& ref, Frame& f)
{
    int Level = mLevel;

    std::vector<cv::Point2f> pts0Raw, pts1Raw;
    std::vector<cv::Point2f> pts0Feature, pts1Feature;
    std::vector<cv::Point2f> pts0, pts1;
    for (int i = 0; i < ref.mKpsPyr[Level].size(); i++) {
        pts0Raw.push_back(ref.mKpsPyr[Level][i].pt);
    }

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(ref.mImgPyr[Level], f.mImgPyr[Level], pts0Raw, pts1Raw, status, err, cv::Size(40,40), 3);
    
    // check the error
    for (int i = 0; i < status.rows; i++) {
        if (status.at<unsigned char>(i, 0) && err.at<unsigned char>(i, 0) < 30) {
            // std::cout << i << ": " << int(err.at<unsigned char>(i, 0)) << std::endl;
            pts0Feature.push_back(pts0Raw[i]);
            pts1Feature.push_back(pts1Raw[i]);
        }
    }

    // check the matching by orb
    pts0 = pts0Feature;
    pts1 = pts1Feature;
    
    // run ransac homography;
    std::vector<int> indexsInPlane;
    cv::Mat HMainPlane = ComputeHomographyFromMatchedPoints(pts0, pts1, indexsInPlane);
    cv::Mat mainPlane  = ExtractPlaneFromHomographyAndRT(HMainPlane, ref.mR, ref.mt, f.mR, f.mt); 

    // visualization. for debug
    #ifndef __ANDROID__
    std::vector<cv::Point2f> pts0Plane, pts1Plane;
    for (int i = 0; i < indexsInPlane.size(); i++) {
        pts0Plane.push_back(pts0[indexsInPlane[i]]);
        pts1Plane.push_back(pts1[indexsInPlane[i]]);
    }
    cv::Mat img4Show = cv::Mat::zeros(ref.mImgPyr[Level].rows, ref.mImgPyr[Level].cols*2, CV_8UC1);
    ref.mImgPyr[Level].copyTo(img4Show(cv::Rect(0, 0, ref.mImgPyr[Level].cols, ref.mImgPyr[Level].rows)));
    f.mImgPyr[Level].copyTo(img4Show(cv::Rect(ref.mImgPyr[Level].cols, 0, ref.mImgPyr[Level].cols, ref.mImgPyr[Level].rows)));
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);
    // std::cout << pts0Plane.size() << std::endl;
    for (int i = 0; i < pts0Plane.size(); i++) {
        cv::circle(img4Show, pts0Plane[i], 3, cv::Scalar(255, 0, 0), 1);
        cv::circle(img4Show, pts1Plane[i]+cv::Point2f(ref.mImgPyr[Level].cols,0), 3, cv::Scalar(255, 0, 255), 1);
        cv::line(img4Show, pts0Plane[i], pts1Plane[i]+cv::Point2f(ref.mImgPyr[Level].cols,0), cv::Scalar(0, 255, 0), 1, CV_AA);
    }
    cv::imshow("image alignment", img4Show);
    #endif

}

cv::Mat PlaneDetector::ComputeHomographyFromMatchedPoints(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, std::vector<int>& indexsPlane)
{
    // play the RANSAC method to find the H stands for the main plane.
    int pointNum = pts0.size();
    std::vector<int> indexs(pointNum);
    for (int i = 0; i < pointNum; i++) {
        indexs[i] = i;
    }

    int pairNum = indexs.size() / 4;
    int iterNum = 10;
    int maxCountHomo = 20;
    std::vector<cv::Mat> HSet(pairNum*iterNum);
    int count = 0;
    for (int iter = 0; iter < iterNum; iter++) {
        std::random_shuffle(indexs.begin(),indexs.end());
        for (int i = 0; i < pairNum; i++) {
            std::vector<cv::Point2f> pts_l;
            pts_l.push_back(pts0[indexs[4*i]]);
            pts_l.push_back(pts0[indexs[4*i+1]]);
            pts_l.push_back(pts0[indexs[4*i+2]]);
            pts_l.push_back(pts0[indexs[4*i+3]]);
            std::vector<cv::Point2f> pts_r;
            pts_r.push_back(pts1[indexs[4*i]]);
            pts_r.push_back(pts1[indexs[4*i+1]]);
            pts_r.push_back(pts1[indexs[4*i+2]]);
            pts_r.push_back(pts1[indexs[4*i+3]]);
            cv::Mat H = cv::findHomography(pts_l, pts_r);
            H.convertTo(H, CV_32FC1);
            // std::cout << H << std::endl;
            float maxerrf = 0.;
            std::vector<float> errvec4 = CheckHomographyReprojError(H, pts_l, pts_r);
            for (int i = 0; i < errvec4.size(); i++) {
                if (errvec4[i] > maxerrf) {
                    maxerrf = errvec4[i];
                }
            }
            // std::cout << "maxerrf: " << maxerrf << std::endl;
            if (maxerrf < 0.99) {
                HSet[count] = H.clone();
                count++;
            }

            if (count > maxCountHomo) {
                break;
            }
        }
        if (count > maxCountHomo) {
            break;
        }
    }
    HSet.resize(count);
    std::cout << "HSet.size(): " << HSet.size() << std::endl;
    
    // voting:
    std::vector<int> supportHSet(HSet.size(), 0);
    std::vector<std::vector<int> > supportPoints(HSet.size());
    for (int i = 0; i < pointNum; i++) {
        int minErrIndex = -1;
        float minErr = 999999.;
        for (int j = 0; j < HSet.size(); j++) {
            // std::cout << HSet[j] << std::endl;
            std::vector<cv::Point2f> im0(1, pts0[i]);
            std::vector<cv::Point2f> im1(1, pts1[i]);
            std::vector<float> err = CheckHomographyReprojError(HSet[j], im0, im1);

            if (err[0] < 1.99) {
                supportHSet[j] ++;
                supportPoints[j].push_back(i);
            }

            // std::cout << err[0] << std::endl;
            // if ( err[0] < minErr ) {
            //     minErr = err[0];
            //     minErrIndex = j;
            // }
        }

        // if (minErr < 0.99) {
        //     supportHSet[minErrIndex] ++;
        // }
        // std::cout << i << ": " << minErr << ", " << minErrIndex << std::endl;
    }

    // std::cout << "supportHSet.size(): " << supportHSet.size() << std::endl;
    int bestPlaneCandidate = 0;
    int maxSupporterNum = 0;
    for (int i = 0; i < supportHSet.size(); i++) {
        // std::cout << supportHSet[i] << " " << std::endl;
        if (supportHSet[i] > maxSupporterNum) {
            bestPlaneCandidate = i;
            maxSupporterNum = supportHSet[i];
        }
    }
    indexsPlane = supportPoints[bestPlaneCandidate];
    // std::cout << bestPlaneCandidate << " " << maxSupporterNum << " " << supportPoints[bestPlaneCandidate].size() << " " << indexsPlane.size() << std::endl;

    return HSet[bestPlaneCandidate];
}

std::vector<float> PlaneDetector::CheckHomographyReprojError(cv::Mat H, std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1)
{
    std::vector<float> res (pts0.size(), 0);

    for (int i = 0; i < pts0.size(); i++) {
        cv::Mat pt0 = (cv::Mat_<float>(3, 1) << pts0[i].x, pts0[i].y, 1.0);
        cv::Mat pt1 = (cv::Mat_<float>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat ptr1 = H*pt0;
        cv::Mat ptr0 = H.inv()*pt1;
        cv::Mat err01 = ptr1/ptr1.at<float>(2,0) - pt1;
        cv::Mat err10 = ptr0/ptr0.at<float>(2,0) - pt0;
        // std::cout << pt0.t() << ptr0.t() << pt1.t() << ptr1.t() << err01.t() << err10.t() << std::endl;
        // res[i] = ( cv::norm(err01) + cv::norm(err10) ) * 0.50;
        res[i] = std::max( cv::norm(err01), cv::norm(err10) );
    }

    return res;
}

cv::Mat PlaneDetector::ExtractPlaneFromHomographyAndRT(cv::Mat H01, cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1)
{
    cv::Mat plane = cv::Mat::ones(4, 1, CV_32FC1);
    cv::Mat R01 = R1*(R0.t());
    cv::Mat t01 = -R1*R0.t()*t0+t1;
    std::cout << R0 << t0.t() << std::endl;
    std::cout << R1 << t1.t() << std::endl;
    std::cout << R01 << t01.t() << std::endl;

    return plane;
}

float PlaneDetector::GetPatchIntense(float u, float v, int width, unsigned char* image)
{
    double res = 0.0;
    for (int dv = -3; dv <= 4; dv++) {
        for (int du = -3; du <= 4; du++) {
            res += float(image[int(u)+du+(int(v)+dv)*width]);
        }
    }
    return res;
}




