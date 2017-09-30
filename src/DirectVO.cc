#include "DirectVO.h" 
#define LEVEL 1
#define FRAMENUM 30

DirectVO::DirectVO(CameraIntrinsic* K)
{
    mK = K;
    mState = TrackingState::VOID;
    mLevel = LEVEL;
    mFrameNum = FRAMENUM;
}

bool DirectVO::SetRefFrame(Frame& f)
{
    // f.ExtractFastPyr();
    f.ExtractGradientPyr(200);
    f.InitDepthPyr(10.0);
    f.ShowPyr(mLevel);

    mRefFrame = f;
    return true;
}

bool DirectVO::AddObvFrame(Frame& f)
{
    mFrameVecBuffer.push_back(f);
    return true;
}

TrackingState DirectVO::TrackMono(cv::Mat image, std::vector<float> R_,  std::vector<float> t_)
{
    Frame f(image);

    if (R_.size() > 0 && t_.size() > 0) {
        std::memcpy(f.mR.ptr<float>(0), R_.data(), R_.size()*sizeof(float));
        std::memcpy(f.mt.ptr<float>(0), t_.data(), t_.size()*sizeof(float));
    }
    // std::cout << f.mR << std::endl << f.mt << std::endl;

    if (mState == TrackingState::VOID) {
        SetRefFrame(f);
        mState = TrackingState::INITIALIZING;
    }
    else if (mState == TrackingState::INITIALIZING) {
        AddObvFrame(f);
        #ifndef __ANDROID__
        // cv::imshow("f", f.mImgPyr[2]);
        #endif
        if (mFrameVecBuffer.size() >= mFrameNum) {
            mState = TrackingState::TRACKING;
        }
    }
    else if (mState == TrackingState::TRACKING) {
        // cv::imshow("mRefFrame", mRefFrame.mImgPyr[0]);
        // cv::imshow("trackFrame0", mFrameVecBuffer[0].mImgPyr[0]);
        BatchOptimizeSE3Depth();
    }

    return mState;
}

bool DirectVO::BatchOptimizeSE3Depth()
{
    int Level = mLevel;
    // Define the problem instance
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // track the points in the level
    float fx = mK->fx / pow(mRefFrame.mScaleFactor, Level);
    float fy = mK->fy / pow(mRefFrame.mScaleFactor, Level);
    float cx = mK->cx / pow(mRefFrame.mScaleFactor, Level);
    float cy = mK->cy / pow(mRefFrame.mScaleFactor, Level);
    int width = mK->width / pow(mRefFrame.mScaleFactor, Level);
    int height = mK->height / pow(mRefFrame.mScaleFactor, Level);
    unsigned char* image0 = mRefFrame.mImgPyr[Level].data;
    cv::Mat se3_0 = mRefFrame.GetDoubleSE3();
    double* pSE3_0 = (double*)se3_0.ptr<double>(0);

    // Init the depth
    // CheckErrorForDistances();
    // set the parameters to optimize: Camera Transformation and Depths
    std::vector<std::vector<double> > SE6Set(mFrameVecBuffer.size(), std::vector<double>(6));
    std::vector<double> depthSet(mRefFrame.mDepthPyr[Level].size());
    for (int i = 0; i < mRefFrame.mDepthPyr[Level].size(); i++) {
        depthSet[i] = double(mRefFrame.mDepthPyr[Level][i]);
    }

    std::cout << "mFrameVecBuffer.size(): " << mFrameVecBuffer.size() << std::endl;
    std::vector<std::vector<double> > SE3Set(mFrameVecBuffer.size());
    for (int n = 0; n < mFrameVecBuffer.size(); n++) {
        unsigned char* imagen = mFrameVecBuffer[n].mImgPyr[Level].data;
        SE3Set[n].resize(6);
        cv::Mat se3_n = mFrameVecBuffer[n].GetDoubleSE3();
        std::memcpy(SE3Set[n].data(), (double*)se3_n.ptr<double>(0), 6*sizeof(double));

        for (int i = 0; i < mRefFrame.mKpsPyr[Level].size(); i++) {
            double u0 = double(mRefFrame.mKpsPyr[Level][i].pt.x);
            double v0 = double(mRefFrame.mKpsPyr[Level][i].pt.y);

            problem.AddResidualBlock(new OptimizationCeres::PhotoMetricErrorDepthCostFuntion(
                u0, v0, width, height, fx, fy, cx, cy, image0, imagen, SE3Set[n].data(), i),
                new ceres::HuberLoss(1.0), &depthSet[i]);
        }
    }

    // To the optimization
    Log_info("Start to solve the problem...");

    // Configure the solver. // for bundle adjustment
    ceres::Solver::Options options;
    options.num_threads = 1;
    options.num_linear_solver_threads = 1;
    options.use_nonmonotonic_steps = false;
    options.use_explicit_schur_complement = true;

    // options.preconditioner_type = ceres::JACOBI;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;

    options.use_inner_iterations = false;
    options.max_num_iterations = 100;
    options.gradient_tolerance = 1e-6;
    options.function_tolerance = 1e-6;

    #ifndef __ANDROID__
        options.minimizer_progress_to_stdout = true;
    #endif

    // Solve!
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "solve done!~\n";
    std::vector<double> residuals;
    double chi2 = 0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), NULL, &residuals, NULL, NULL);

    for (int i = 0; i < residuals.size(); i++) {
        // int n = i / mKpsPyr[Level].size();
        int k = i % mRefFrame.mKpsPyr[Level].size();
        if (residuals[i] != 0) {
            // mRefFrame.mStatisticPyr[Level][k].add(abs(residuals[i]));
        }
    }

    #ifndef __ANDROID__
    std::cout << "Final report:\n" << summary.FullReport();
    #endif

    int N = mFrameNum;
    std::cout << "depthSet.size(): " << depthSet.size() << std::endl;
    for (int i = 0; i < depthSet.size(); i++) {
        // Log_info("depth: {},  {} - {} {} {}", i, depthSet[i], 
        //                                     mRefFrame.mStatisticPyr[Level][i].getMean(),
        //                                     mRefFrame.mStatisticPyr[Level][i].getStdVariance(),
        //                                     mRefFrame.mStatisticPyr[Level][i].getNum());
        mRefFrame.mDepthPyr[Level][i] = float(depthSet[i]);
    }

    CheckErrorForDistances();
    ShowDepth(mRefFrame);
    // for (int i = 0; i < mFrameVecBuffer.size(); i++) {
    //     CheckReprojection(mRefFrame, mFrameVecBuffer[i]);
    //     cv::waitKey(-1);
    // }

    cv::waitKey(-1);
    return true;
}

float DirectVO::GetReprojectionPhotometricError(Frame& ref, Frame& f, cv::Point2f p_ref, float d)
{
    float u0 = p_ref.x;
    float v0 = p_ref.y;

    int Level = mLevel;

    unsigned char* image0 = ref.mImgPyr[Level].data;
    unsigned char* image1 = f.mImgPyr[Level].data;

    float fx = mK->fx / pow(ref.mScaleFactor, Level);
    float fy = mK->fy / pow(ref.mScaleFactor, Level);
    float cx = mK->cx / pow(ref.mScaleFactor, Level);
    float cy = mK->cy / pow(ref.mScaleFactor, Level); 
    int   width  = mK->width  / pow(ref.mScaleFactor, Level); 
    int   height = mK->height / pow(ref.mScaleFactor, Level); 
    // std::cout << fx << " " << fy << " " << cx << " " << cy << " " << width << " " << height << std::endl;
    
    float x, y, z;
    z = d;
    x = (u0 - cx) / fx * z;
    y = (v0 - cy) / fy * z;

    float* R = f.mR.ptr<float>(0);
    float* t = f.mt.ptr<float>(0);

    float u1 = fx * (R[0]*x + R[1]*y + R[2]*z + t[0]) / (R[6]*x + R[7]*y + R[8]*z + t[2]) + cx;
    float v1 = fy * (R[3]*x + R[4]*y + R[5]*z + t[1]) / (R[6]*x + R[7]*y + R[8]*z + t[2]) + cy;

    if (u1 < 5 || u1 > width-5 || v1 < 5 || v1 > height-5) {
        return 9999999.;
    }

    double xx0 = u0 - int(u0);
    double yy0 = v0 - int(v0);
    double WLT0 = (1-xx0)*(1-yy0);
    double WRT0 = xx0*(1-yy0);
    double WLB0 = (1-xx0)*yy0;
    double WRB0 = xx0*yy0;
    // double I0  = GetPatchIntense(u0, v0, width, image0) * WLT0
    //                 + GetPatchIntense(u0+1, v0, width, image0) * WRT0
    //                 + GetPatchIntense(u0, v0+1, width, image0) * WLB0
    //                 + GetPatchIntense(u0+1, v0+1, width, image0) * WRB0;
    double I0  =   image0[int(u0)   + width*int(v0)]   * WLT0 
                 + image0[int(u0)+1 + width*int(v0)]   * WRT0
                 + image0[int(u0)   + width*(int(v0)+1)] * WLB0
                 + image0[int(u0)+1 + width*(int(v0)+1)] * WRB0;

    double xx1 = u1 - int(u1);
    double yy1 = v1 - int(v1);
    double WLT1 = (1-xx1)*(1-yy1);
    double WRT1 = xx1*(1-yy1);
    double WLB1 = (1-xx1)*yy1;
    double WRB1 = xx1*yy1;
    // double I1  = GetPatchIntense(u1, v1, width, image1) * WLT1
    //                 + GetPatchIntense(u1+1, v1, width, image1) * WRT1
    //                 + GetPatchIntense(u1, v1+1, width, image1) * WLB1
    //                 + GetPatchIntense(u1+1, v1+1, width, image1) * WRB1;
    double I1  =   image1[int(u1)   + width*int(v1)]   * WLT1 
                 + image1[int(u1)+1 + width*int(v1)]   * WRT1
                 + image1[int(u1)   + width*int((v1)+1)] * WLB1
                 + image1[int(u1)+1 + width*int((v1)+1)] * WRB1;


    // std::cout << d << "\t" << u0 << "\t" << v0 << "\t" << u1 << "\t" << v1 << "\t" << I0 << "\t" << I1 << "\t" << abs(I0 - I1) << std::endl;

    return abs(I0 - I1);
    
}

void DirectVO::CheckReprojection(Frame& ref, Frame& f)
{
    int Level = mLevel;
    std::vector<cv::Point2f> pts0;
    std::vector<cv::Point2f> pts1;
    float fx = mK->fx / pow(mRefFrame.mScaleFactor, Level);
    float fy = mK->fy / pow(mRefFrame.mScaleFactor, Level);
    float cx = mK->cx / pow(mRefFrame.mScaleFactor, Level);
    float cy = mK->cy / pow(mRefFrame.mScaleFactor, Level);

    for (int i = 0; i < ref.mKpsPyr[Level].size(); i++) {
        pts0.push_back(ref.mKpsPyr[Level][i].pt);
        float x, y, z;
        x = (ref.mKpsPyr[Level][i].pt.x - cx) / fx * ref.mDepthPyr[Level][i];
        y = (ref.mKpsPyr[Level][i].pt.y - cy) / fy * ref.mDepthPyr[Level][i];
        z = ref.mDepthPyr[Level][i];

        float* R = f.mR.ptr<float>(0);
        float* t = f.mt.ptr<float>(0);

        float u1 = fx * (R[0]*x + R[1]*y + R[2]*z + t[0]) / (R[6]*x + R[7]*y + R[8]*z + t[2]) + cx;
        float v1 = fy * (R[3]*x + R[4]*y + R[5]*z + t[1]) / (R[6]*x + R[7]*y + R[8]*z + t[2]) + cy;
        pts1.push_back(cv::Point2f(u1, v1));
    }

    #ifndef __ANDROID__
    cv::Mat img4Show = cv::Mat::zeros(ref.mImgPyr[Level].rows, ref.mImgPyr[Level].cols*2, CV_8UC1);
    ref.mImgPyr[Level].copyTo(img4Show(cv::Rect(0, 0, ref.mImgPyr[Level].cols, ref.mImgPyr[Level].rows)));
    f.mImgPyr[Level].copyTo(img4Show(cv::Rect(ref.mImgPyr[Level].cols, 0, ref.mImgPyr[Level].cols, ref.mImgPyr[Level].rows)));
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);
    for (int i = 0; i < pts0.size(); i++) {
        cv::circle(img4Show, pts0[i], 3, cv::Scalar(255, 0, 0), 1);
        cv::circle(img4Show, pts1[i]+cv::Point2f(ref.mImgPyr[Level].cols,0), 3, cv::Scalar(255, 0, 255), 1);
        cv::line(img4Show, pts0[i], pts1[i]+cv::Point2f(ref.mImgPyr[Level].cols,0), cv::Scalar(0, 255, 0), 1, CV_AA);
    }
    cv::imshow("img align", img4Show);
    #endif
}

void DirectVO::CheckErrorForDistances()
{

    int Level = mLevel;
    for (int p = 0; p < mRefFrame.mKpsPyr[Level].size(); p++) {
        float ep_min = 9999999.;
        float best_d = 10;
        for (float d = 1.0; d < 20.0; d+=1.0) {
            float ep = 0;
            for (int i = 0; i < mFrameVecBuffer.size(); i++) {
                ep += GetReprojectionPhotometricError(mRefFrame, mFrameVecBuffer[i], mRefFrame.mKpsPyr[Level][p].pt, d);
            }
            std::cout << ep << " ";
            if (ep < ep_min)
            {
                ep_min = ep;
                best_d = d;
            }
        }
        
        std::cout << "  (" << best_d << " " << ep_min << ")" << std::endl;
        if (ep_min < 3 * mFrameVecBuffer.size())
            mRefFrame.mDepthPyr[Level][p] = best_d;
        else 
            mRefFrame.mDepthPyr[Level][p] = 0;
    }
    
}

void DirectVO::ShowDepth(Frame& f)
{
    int Level = mLevel;
    cv::Mat imgDepth = cv::Mat::zeros(f.mImgPyr[Level].rows, f.mImgPyr[Level].cols, CV_32FC1);
    for (int i = 0; i < f.mDepthPyr[Level].size(); i++) {
        float d = f.mDepthPyr[Level][i];
        // std::cout << "d " << i << ": " << d << std::endl;
        if (d > 25)
            d = 0;
        else if (d < 5)
            d = 0;
        imgDepth.at<float>(int(f.mKpsPyr[Level][i].pt.y), int(f.mKpsPyr[Level][i].pt.x)) 
                = 1.0 - (d - 5.0) / 25.0;
    }
    cv::imshow("depth", imgDepth);
}

double DirectVO::GetPatchIntense(float u, float v, int width, unsigned char* image)
{
    double res = 0.0;
    for (int dv = -3; dv <= 4; dv++) {
        for (int du = -3; du <= 4; du++) {
            res += double(image[int(u)+du+(int(v)+dv)*width]);
        }
    }
    return res;
}