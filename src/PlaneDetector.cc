#include "PlaneDetector.h" 
#define LEVEL 0
#define FRAMENUM 5

PlaneDetector::PlaneDetector(CameraIntrinsic* K)
{
    mK = K;
    mState = PlaneDetectionState::VOID;
    mLevel = LEVEL;
    mFrameNum = FRAMENUM;
    mWinnerTextureID = -1;
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

PlaneDetectionState PlaneDetector::Detect(cv::Mat image, std::vector<float> R_,  std::vector<float> t_)
{
    Frame f(image);

    if (R_.size() > 0 && t_.size() > 0) {
        std::memcpy(f.mR.ptr<float>(0), R_.data(), R_.size()*sizeof(float));
        std::memcpy(f.mt.ptr<float>(0), t_.data(), t_.size()*sizeof(float));
    }

    if (mState == PlaneDetectionState::VOID) {
        SetRefFrame(f);
        TIME_BEGIN();
        mTextureSeg.InitData(mRefFrame.mImgPyr[mLevel], 20, 20);
        mTextureSeg.ComputeGridFeatures();
        TIME_END("Segment Ref Frame Image ");
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
        DetectMatchByOpticalFlow(mRefFrame, mFrameVecBuffer[2]);
        TIME_END("DetectMatchByOpticalFlow");
        mState = PlaneDetectionState::FILTERING;
    }
    else if (mState == PlaneDetectionState::FILTERING) {

        bool ifExtractPlane = 0;
        std::vector<int> indexPtsSetAsPlaneCandidate;
        cv::Mat F_S_G, F_T_G, F_T, F_S, F_S_T;
        TIME_BEGIN();
        F_S_G = CalculateConditionalDistribution_SurfaceGrid(mPixelsMatchHMatrixSurfaceOnRefFrame);
        F_T_G = CalculateConditionalDistribution_TextureGrid();
        F_T   = CalculateMarginalDistribution_Texture(F_T_G);
        F_S   = CalculateMarginalDistribution_Surface(F_S_G);
        F_S_T = CalculateConditionalDistribution_SurfaceTexture(F_S_G, F_T_G);
        TIME_END("Calculate prob distributions.");

        ifExtractPlane = UpdatePlaneByTextureRelatedPoints(mPixelsMatchHMatrixSurfaceOnRefFrame, F_T_G, F_S_T, indexPtsSetAsPlaneCandidate);

        if (ifExtractPlane) {
            mState = PlaneDetectionState::END;
        } else {
            Log_info("Failed to extract the plane.");
            mState = PlaneDetectionState::FAILED;
        }
    }
    else if (mState == PlaneDetectionState::FAILED ){
        Log_info("Go to the failed state.");
    }
    else if (mState == PlaneDetectionState::END ){
        // Viewer::DrawAR(mK, f, mMainPlane);
        Viewer::DrawAR(mK, f, mMainPlane, mAnchorPoint);
        std::vector<cv::Point3f> planeGrids = GetPlaneRegionUsingAnchorPointAndTexture();
        Viewer::DrawPlane(mK, f, mMainPlane, planeGrids);
        cv::waitKey(-1);
    }

    return mState;
}

void PlaneDetector::DetectMatchByOpticalFlow(Frame& ref, Frame& f)
{
    int Level = mLevel;
    cv::Mat imageref = ref.mImgPyr[Level];
    cv::Mat imagef   = f.mImgPyr[Level];
    std::vector<cv::KeyPoint> kpsref = ref.mKpsPyr[Level];

    std::vector<int>         indexPtsRaw, indexPtsFeature, indexPts, indexPtsPlane;
    std::vector<cv::Point2f> pts0Raw, pts1Raw;
    std::vector<cv::Point2f> pts0Feature, pts1Feature;
    std::vector<cv::Point2f> pts0, pts1;
    std::vector<cv::Point2f> pts0Plane, pts1Plane;

    for (int i = 0; i < kpsref.size(); i++) {
        pts0Raw.push_back(kpsref[i].pt);
        indexPtsRaw.push_back(i);
    }

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(imageref, imagef, pts0Raw, pts1Raw, status, err, cv::Size(20,20), 3);
    
    // check the error
    for (int i = 0; i < status.rows; i++) {
        if (status.at<unsigned char>(i, 0) && err.at<unsigned char>(i, 0) < 30) {
            // std::cout << i << ": " << int(err.at<unsigned char>(i, 0)) << std::endl;
            indexPtsFeature.push_back(indexPtsRaw[i]);
            pts0Feature.push_back(pts0Raw[i]);
            pts1Feature.push_back(pts1Raw[i]);
        }
    }

    // check the matching by orb
    indexPts = indexPtsFeature;
    pts0     = pts0Feature;
    pts1     = pts1Feature;
    
    // run ransac homography and select the point pairs inside the region;
    std::vector<int> indexsInPlane;
    cv::Mat HMainPlane = ComputeHomographyFromMatchedPoints(pts0, pts1, indexsInPlane);
    for (int i = 0; i < indexsInPlane.size(); i++) {
        indexPtsPlane.push_back(indexPts[indexsInPlane[i]]);
        pts0Plane.push_back(pts0[indexsInPlane[i]]);
        pts1Plane.push_back(pts1[indexsInPlane[i]]);
    }

    RecoverPlaneFromPointPairsAndRT(pts0Plane, pts1Plane, ref.mR, ref.mt, f.mR, f.mt, 
                                    mPixelsMatchHMatrixSurfaceOnRefFrame,
                                    mPoints3DMatchHMatrixSurface, 
                                    mMainPlane, mAnchorPoint);

    // visualization. for debug
    #ifndef __ANDROID__
    cv::Mat img4Show = cv::Mat::zeros(imageref.rows, imageref.cols*2, CV_8UC1);
    imageref.copyTo(img4Show(cv::Rect(0, 0, imageref.cols, imageref.rows)));
    imagef.copyTo(img4Show(cv::Rect(imageref.cols, 0, imageref.cols, imageref.rows)));
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);
    // std::cout << pts0Plane.size() << std::endl;
    for (int i = 0; i < indexPtsPlane.size(); i++) {
        cv::circle(img4Show, pts0Raw[indexPtsPlane[i]], 3, cv::Scalar(255, 0, 0), 1);
        cv::circle(img4Show, pts1Raw[indexPtsPlane[i]]+cv::Point2f(imageref.cols,0), 3, cv::Scalar(255, 0, 255), 1);
        cv::line(img4Show, pts0Raw[indexPtsPlane[i]], pts1Raw[indexPtsPlane[i]]+cv::Point2f(imageref.cols,0), cv::Scalar(0, 255, 0), 1, CV_AA);
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
    int iterNum = 20;
    int maxCountHomo = 30;
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
    // std::cout << "HSet.size(): " << HSet.size() << std::endl;
    
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

            if (err[0] < 1.49) {
                supportHSet[j] ++;
                supportPoints[j].push_back(i);
            }
        }
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

bool PlaneDetector::RecoverPlaneFromHomographyAndRT(cv::Mat H01, cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1)
{

    cv::Mat R01 = R1*(R0.t());
    cv::Mat t01 = -R1*R0.t()*t0+t1;
    std::cout << R0 << t0.t() << std::endl;
    std::cout << R1 << t1.t() << std::endl;
    std::cout << R01 << t01.t() << std::endl;

    return true;
}

bool PlaneDetector::RecoverPlaneFromPointPairsAndRT(std::vector<cv::Point2f> pts0, std::vector<cv::Point2f> pts1, 
                                                       cv::Mat R0, cv::Mat t0, cv::Mat R1, cv::Mat t1,
                                                       std::vector<cv::Point2f>& pixels3d,
                                                       std::vector<cv::Point3f>& points3d,
                                                       std::vector<float>& mainPlane, std::vector<float>& anchorPoint)
{
    cv::Mat plane = cv::Mat::ones(4, 1, CV_32FC1);
    cv::Mat Kh = (cv::Mat_<float>(3, 3) << mK->fx, 0, mK->cx, 0, mK->fy, mK->cy, 0, 0, 1);
    cv::Mat T0, T1, p4d;
    cv::hconcat(R0, t0, T0);
    cv::hconcat(R1, t1, T1);
    // std::cout << Kh << std::endl << T0 << std::endl << T1 << std::endl;
    T0 = Kh * T0;
    T1 = Kh * T1;
    cv::triangulatePoints(T0, T1, pts0, pts1, p4d);
    p4d.convertTo(p4d, CV_32FC1);
    for (int i = 0; i < p4d.cols; i++) {
        if (p4d.at<float>(3,i) != 0) {
            p4d.at<float>(0,i) /= p4d.at<float>(3,i);       
            p4d.at<float>(1,i) /= p4d.at<float>(3,i); 
            p4d.at<float>(2,i) /= p4d.at<float>(3,i);  
            p4d.at<float>(3,i) /= p4d.at<float>(3,i);  
            if ( p4d.at<float>(2,i) <= 0.000001 ) {
                p4d.at<float>(3,i) = 0;
            }    
        }
        // std::cout << pts0[i] << p4d.col(i).t() << std::endl; 
    }

    std::vector<cv::Point2f> pixels;
    std::vector<cv::Point3f> points;
    for (int i = 0; i < p4d.cols; i++) {
        if (p4d.at<float>(3,i) != 0) {
            points.push_back(cv::Point3f(p4d.at<float>(0,i), p4d.at<float>(1,i), p4d.at<float>(2,i)));
            pixels.push_back(pts0[i]);
        }
    }
    pixels3d = pixels;
    points3d = points;

    // std::cout << points <<std::endl;
    float planeConfidence = RecoverPlaneFrom3DPoints(points, mainPlane, anchorPoint);

    return true;
}

bool PlaneDetector::UpdatePlaneByTextureRelatedPoints(std::vector<cv::Point2f> pts, cv::Mat F_T_G, cv::Mat F_S_T, std::vector<int>& indexPts)
{
    std::vector<int>   grids;
    cv::Mat weights(pts.size(), 1, CV_32FC1);

    int gridX = mTextureSeg.mGridX;
    int gridY = mTextureSeg.mGridY;
    int gridNumX = mTextureSeg.mGridNumX;
    
    for (int i = 0; i < pts.size(); i++) {
        int x = pts[i].x / gridX;
        int y = pts[i].y / gridY;
        int id = x + y * gridNumX;
        grids.push_back(id);

        cv::Mat f_t_g = F_T_G.row(id);
        cv::Mat f_s_t = F_S_T.col(1);
        // std::cout << f_t_g << f_s_t << std::endl;
        cv::Mat weightMat = f_t_g * f_s_t;
        weights.at<float>(i, 0) = weightMat.at<float>(0, 0);
        std::cout << pts[i] << x << " " << y << " " << f_t_g << f_s_t.t() << weights.at<float>(i, 0) << std::endl;
    }

    double minw, maxw;
    cv::minMaxLoc(weights, &minw, &maxw);
    float  th1 = float(maxw) * 0.95;
    float  th2 = float(maxw) * 0.80;
    
    std::vector<int> indexes;
    std::vector<int> gridsIndexes;
    for (int i = 0; i < weights.rows; i++) {
        if (weights.at<float>(i, 0) >= th1) {
            indexes.push_back(i);
            gridsIndexes.push_back(grids[i]);
        }
    }
    if (indexes.size() < 0.3 * weights.rows) {
        for (int i = 0; i < weights.rows; i++) {
            if (weights.at<float>(i, 0) < th1 && weights.at<float>(i, 0) >= th2) {
                indexes.push_back(i);
                gridsIndexes.push_back(grids[i]);
            }
        }       
    }

    int candidatesSize = indexes.size();
    if (candidatesSize > 5) {
        std::vector<cv::Point3f> pt3dCandidates;
        std::vector<float> textureWeights(F_S_T.rows, 0);
        for (int i = 0; i < indexes.size(); i++) {
            pt3dCandidates.push_back(mPoints3DMatchHMatrixSurface[indexes[i]]);
            for (int j = 0; j < textureWeights.size(); j++) {
                textureWeights[j] += F_T_G.at<float>(gridsIndexes[i], j);
            }
        }

        int targetTextureID = -1;
        int mostTextureID = 0;
        for (int i = 0; i < textureWeights.size(); i++) {
            std::cout << textureWeights[i] << " ";
            if (textureWeights[i] > mostTextureID) {
                mostTextureID   = textureWeights[i];
                targetTextureID = i;
            }
        } std::cout << std::endl << "targetTextureID: " << targetTextureID << std::endl;
        mWinnerTextureID = targetTextureID;

        RecoverPlaneFrom3DPoints(pt3dCandidates, mMainPlane, mAnchorPoint);
        return true;
    }

    return false;
}

std::vector<cv::Point3f> PlaneDetector::GetPlaneRegionUsingAnchorPointAndTexture()
{
    std::vector<float> mainPlane   = mMainPlane;
    std::vector<float> anchorPoint = mAnchorPoint;
    int winnerTextureID = mWinnerTextureID;

    cv::Mat Rref = mRefFrame.mR;
    cv::Mat tref = mRefFrame.mt;

    int gridX = mTextureSeg.mGridX;
    int gridY = mTextureSeg.mGridY;
    int gridNumX = mTextureSeg.mGridNumX;
    int gridNumY = mTextureSeg.mGridNumY;

    float a = mainPlane[0];
    float b = mainPlane[1];
    float c = mainPlane[2];
    float d = mainPlane[3];
    
    std::vector<cv::Point3f> targetGridsProjections;

    for (int y = 0; y < gridNumY; y++) {
        for (int x = 0; x < gridNumX; x++) {
            if (mTextureSeg.mTextureMap.at<int>(y, x) == winnerTextureID) {
                cv::Point2f gridCenter(x*gridX+gridX/2, y*gridY+gridY/2);

                float u = gridCenter.x;
                float v = gridCenter.y;
                // normal line: 
                float nxref = (u - mK->cx) / mK->fx;
                float nyref = (v - mK->cy) / mK->fy;
                float nzref = 1;
                cv::Mat nxyzref = (cv::Mat_<float>(3,1) << nxref, nyref, nzref);
                cv::Mat nxyz = Rref.t() * nxyzref;
                float nx = nxyz.at<float>(0,0);
                float ny = nxyz.at<float>(1,0);
                float nz = nxyz.at<float>(2,0);
                // origin point:
                cv::Mat origin = -tref;
                float ox = origin.at<float>(0,0);
                float oy = origin.at<float>(1,0);
                float oz = origin.at<float>(2,0);

                // get the intersection the line to the plane.
                float depth = ( -d-(a*ox+b*oy+c*oz) ) / ( a*nx+b*ny+c*nz );
                cv::Point3f intersect(ox+nx*depth, oy+ny*depth, oz+nz*depth);
                targetGridsProjections.push_back(intersect);
            }
        }
    }

    return targetGridsProjections;

}


// F(S|G).
cv::Mat PlaneDetector::CalculateConditionalDistribution_SurfaceGrid(std::vector<cv::Point2f> ptsMatchHMat)
{
    cv::Mat p_surface_grid = cv::Mat::zeros(mTextureSeg.mTextureMap.cols*mTextureSeg.mTextureMap.rows, 2, CV_32FC1);

    std::vector<cv::Point2f> pts = ptsMatchHMat;
    float R = 0.36 * cv::norm(cv::Point2f(mTextureSeg.mGridX/2, mTextureSeg.mGridY/2));

    for (int y = 0; y < mTextureSeg.mTextureMap.rows; y++) {
        for (int x = 0; x < mTextureSeg.mTextureMap.cols; x++) {
            int id = x + y * mTextureSeg.mTextureMap.cols;
            cv::Point2f cGrid(x*mTextureSeg.mGridX + mTextureSeg.mGridX/2, y*mTextureSeg.mGridY+mTextureSeg.mGridY/2);
            float d, score = 0.0;
            for (int i = 0; i < pts.size(); i++) {
                d = cv::norm(pts[i] - cGrid)/R;
                if (std::exp(-d) > score) {
                    score = std::exp(-d/R);
                }
            }
            p_surface_grid.at<float>(id, 1) = score;
            p_surface_grid.at<float>(id, 0) = 1. - score;
        }
    }

    #ifndef __ANDROID__
    cv::Mat img_p_surface_grid = cv::Mat::zeros(mTextureSeg.mTextureMap.rows, mTextureSeg.mTextureMap.cols, CV_32FC1);
    for (int y = 0; y < mTextureSeg.mTextureMap.rows; y++) {
        for (int x = 0; x < mTextureSeg.mTextureMap.cols; x++) {
            img_p_surface_grid.at<float>(y, x) = p_surface_grid.at<float>(y*mTextureSeg.mTextureMap.cols+x, 1);
        }
    }
    cv::imshow("img_p_surface_grid", img_p_surface_grid);
    // std::cout << p_surface_grid << std::endl;
    #endif
    return p_surface_grid;
}

// F(S|T) 
cv::Mat PlaneDetector::CalculateConditionalDistribution_SurfaceTexture(cv::Mat F_S_G, cv::Mat F_T_G)
{   
    cv::Mat p_surface_texture = cv::Mat::zeros(mTextureSeg.mTextureID+1, 2, CV_32FC1);

    // std::cout << F_S_G << std::endl << F_T_G << std::endl;

    p_surface_texture = (F_S_G.t() * F_T_G).t();

    p_surface_texture.col(0) = p_surface_texture.col(0) / cv::sum(p_surface_texture.col(0)).val[0];
    p_surface_texture.col(1) = p_surface_texture.col(1) / cv::sum(p_surface_texture.col(1)).val[0];
    
    float p0, p1;
    for (int t = 0; t < p_surface_texture.rows; t++) {
        p0 = p_surface_texture.at<float>(t, 0);
        p1 = p_surface_texture.at<float>(t, 1);
        p_surface_texture.at<float>(t, 0) = p_surface_texture.at<float>(t, 0) * p0 / (p0 + p1);
        p_surface_texture.at<float>(t, 1) = p_surface_texture.at<float>(t, 1) * p1 / (p0 + p1);
    }

    std::cout << "p_surface_texture: " << std::endl << p_surface_texture << std::endl;

    return p_surface_texture;
}

// F(G|T)
cv::Mat PlaneDetector::CalculateConditionalDistribution_TextureGrid()
{
    cv::Mat p_texture_grid = cv::Mat::zeros(mTextureSeg.mTextureMap.cols*mTextureSeg.mTextureMap.rows, mTextureSeg.mTextureID+1, CV_32FC1);

    int dx[4] = {-1, 1,  0, 0};
    int dy[4] = { 0, 0, -1, 1};
    std::cout << "mTextureSeg.mTextureID: " << mTextureSeg.mTextureID << std::endl;
    std::cout << mTextureSeg.mTextureMap << std::endl;
    for (int y = 1; y < mTextureSeg.mTextureMap.rows-1; y++) {
        for (int x = 1; x < mTextureSeg.mTextureMap.cols-1; x++) {
            if (mTextureSeg.mTextureMap.at<int>(y, x) >= 0) {
                int id = x + y * mTextureSeg.mTextureMap.cols;
                std::vector<float> res(mTextureSeg.mTextureID+1, 0);
                res[mTextureSeg.mTextureMap.at<int>(y, x)]++;
                for (int i = 0; i < 4; i++) {
                    int classID = mTextureSeg.mTextureMap.at<int>(y+dy[i], x+dx[i]);
                    if (classID >= 0) {
                        res[classID]++;
                    }
                }
                int resSum = 0;
                for (int i = 0; i < res.size(); i++) {
                    resSum+=res[i];
                }
                for (int i = 0; i < res.size(); i++) {
                    res[i] /= resSum;
                    p_texture_grid.at<float>(id, i) = res[i];
                }               
            }
        }
    }

    // std::cout << p_texture_grid << std::endl;
    return p_texture_grid;
}

// F(T)
cv::Mat PlaneDetector::CalculateMarginalDistribution_Texture(cv::Mat F_T_G)
{
    cv::Mat p_texture = cv::Mat::zeros(mTextureSeg.mTextureID+1, 1, CV_32FC1);

    int G = F_T_G.rows;
    for (int g = 0; g < G; g++) {
        for (int t = 0; t <= mTextureSeg.mTextureID; t++) {
            p_texture.at<float>(t, 0) += F_T_G.at<float>(g, t);
        }
    }
    p_texture = p_texture / G;
    std::cout << p_texture.t() << std::endl;

    return p_texture;
}

// F(S)
cv::Mat PlaneDetector::CalculateMarginalDistribution_Surface(cv::Mat F_S_G)
{
    cv::Mat p_surface = cv::Mat::zeros(2, 1, CV_32FC1);

    int Ng = F_S_G.rows;
    for (int g = 0; g < Ng; g++) {
        p_surface.at<float>(0, 0) += F_S_G.at<float>(g, 0);
        p_surface.at<float>(1, 0) += F_S_G.at<float>(g, 1);
    }

    float f0 = p_surface.at<float>(0, 0);
    float f1 = p_surface.at<float>(0, 1);
    p_surface.at<float>(0, 0) *= 1. / (f0 + f1);
    p_surface.at<float>(0, 1) *= 1. / (f0 + f1);

    std::cout << p_surface.t() << std::endl;
    return p_surface;
}

float PlaneDetector::RecoverPlaneFrom3DPoints(std::vector<cv::Point3f> p3ds, std::vector<float>& mainPlane, std::vector<float>& anchorPoint)
{
    float res = 1.0;

    int pointNum = p3ds.size();

    std::cout << "pointNum: " << pointNum << std::endl;
    // find the mean points of the point cloud
    cv::Point3f c(0, 0, 0);
    for (int i = 0; i < pointNum; i++) {
        c = c + p3ds[i];
    }
    c = c / pointNum;  

    // normalize the point cloud by the mean point
    cv::Mat p3Mat(pointNum, 3, CV_32FC1);
    for (int i = 0; i < pointNum; i++) {
        p3Mat.at<float>(i, 0) = p3ds[i].x - c.x;
        p3Mat.at<float>(i, 1) = p3ds[i].y - c.y;
        p3Mat.at<float>(i, 2) = p3ds[i].z - c.z;
    }

    // std::cout << p3Mat << std::endl;

    // calculate the covariance matrix of the 3 dimensions
    cv::Mat covMat(3, 3, CV_32FC1);
    // xx
    covMat.at<float>(0, 0) = cv::sum(p3Mat.col(0).mul(p3Mat.col(0))).val[0];
    // yy
    covMat.at<float>(1, 1) = cv::sum(p3Mat.col(1).mul(p3Mat.col(1))).val[0];
    // zz
    covMat.at<float>(2, 2) = cv::sum(p3Mat.col(2).mul(p3Mat.col(2))).val[0];
    // xy
    covMat.at<float>(0, 1) = cv::sum(p3Mat.col(0).mul(p3Mat.col(1))).val[0];
    covMat.at<float>(1, 0) = cv::sum(p3Mat.col(1).mul(p3Mat.col(0))).val[0];
    // xz
    covMat.at<float>(0, 2) = cv::sum(p3Mat.col(0).mul(p3Mat.col(2))).val[0];
    covMat.at<float>(2, 0) = cv::sum(p3Mat.col(2).mul(p3Mat.col(0))).val[0];
    // yz
    covMat.at<float>(1, 2) = cv::sum(p3Mat.col(1).mul(p3Mat.col(2))).val[0];
    covMat.at<float>(2, 1) = cv::sum(p3Mat.col(2).mul(p3Mat.col(1))).val[0];

    std::cout << "covMat: " << std::endl << covMat << std::endl;
    cv::SVD svd;
    cv::Mat w, u, vt;
    svd.compute(covMat, w, u, vt);
    std::cout << u << std::endl << w << std::endl << vt << std::endl;
    cv::Mat normal = vt.t().col(2);
    float d = -( c.x*normal.at<float>(0,0) + c.y*normal.at<float>(1,0) + c.z*normal.at<float>(2,0) );
    std::cout << normal.t() << d << std::endl;
    normal = normal / d;
    d = d / d;

    mainPlane.resize(4);
    mainPlane[0] = normal.at<float>(0,0);
    mainPlane[1] = normal.at<float>(1,0);
    mainPlane[2] = normal.at<float>(2,0);
    mainPlane[3] = d;

    anchorPoint.resize(3);
    anchorPoint[0] = c.x;
    anchorPoint[1] = c.y;
    anchorPoint[2] = c.z;

    return res;
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




