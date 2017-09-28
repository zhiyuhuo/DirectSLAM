#include "DirectVO.h" 
#define LEVEL 1

DirectVO::DirectVO(CameraIntrinsic* K)
{
    mK = K;
    mState = TrackingState::VOID;
}

bool DirectVO::SetRefFrame(Frame& f)
{
    //f.ExtractFastPyr();
    f.ExtractSlopePyr(100);
    f.InitDepthPyr(10.0);
    f.ShowPyr(LEVEL);

    mRefFrame = f;
    return true;
}

bool DirectVO::AddObvFrame(Frame& f)
{
    // std::cout << f.mR << std::endl;
    // std::cout << f.mt << std::endl;
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
    std::cout << f.mR << std::endl << f.mt << std::endl;

    if (mState == TrackingState::VOID) {
        SetRefFrame(f);
        mState = TrackingState::INITIALIZING;
    }
    else if (mState == TrackingState::INITIALIZING) {
        AddObvFrame(f);
        #ifndef __ANDROID__
        cv::imshow("f", f.mImgPyr[2]);
        #endif
        if (mFrameVecBuffer.size() >= 20) {
            mState = TrackingState::TRACKING;
        }
    }
    else if (mState == TrackingState::TRACKING) {
        BatchOptimizeSE3Depth(1);
    }

    return mState;
}

bool DirectVO::BatchOptimizeSE3Depth(int levelNum)
{
    // Define the problem instance
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // track the points in the highest level: 3
    float fx = mK->fx / pow(mRefFrame.mScaleFactor, LEVEL);
    float fy = mK->fy / pow(mRefFrame.mScaleFactor, LEVEL);
    float cx = mK->cx / pow(mRefFrame.mScaleFactor, LEVEL);
    float cy = mK->cy / pow(mRefFrame.mScaleFactor, LEVEL);
    int width = mK->width / pow(mRefFrame.mScaleFactor, LEVEL);
    int height = mK->height / pow(mRefFrame.mScaleFactor, LEVEL);
    unsigned char* image0 = mRefFrame.mImgPyr[LEVEL].data;
    cv::Mat se3_0 = mRefFrame.GetDoubleSE3();
    double* pSE3_0 = (double*)se3_0.ptr<double>(0);

    // set the parameters to optimize: Camera Transformation and Depths
    std::vector<std::vector<double> > SE6Set(mFrameVecBuffer.size(), std::vector<double>(6));
    std::vector<double> depthSet(mRefFrame.mDepthPyr[LEVEL].size());
    for (int i = 0; i < mRefFrame.mDepthPyr[LEVEL].size(); i++) {
        depthSet[i] = double(mRefFrame.mDepthPyr[LEVEL][i]);
    }

    // for (int n = 0; n < mFrameVecBuffer.size(); n++) {
    //     unsigned char* imagen = mFrameVecBuffer[n].mImgPyr[LEVEL].data;
    //     for (int i = 0; i < mRefFrame.mKpsPyr[LEVEL].size(); i++) {
    //         double u0 = double(mRefFrame.mKpsPyr[LEVEL][i].pt.x);
    //         double v0 = double(mRefFrame.mKpsPyr[LEVEL][i].pt.y);

    //         problem.AddResidualBlock(new OptimizationCeres::PhotoMetricErrorSE3DepthCostFuntion(
    //             u0, v0, width, height, fx, fy, cx, cy, image0, imagen, i),
    //             new ceres::HuberLoss(1.0), SE6Set[n].data(), &depthSet[i]);
    //     }
    // }

    std::vector<std::vector<double> > SE3Set(mFrameVecBuffer.size());
    for (int n = 0; n < mFrameVecBuffer.size(); n++) {
        unsigned char* imagen = mFrameVecBuffer[n].mImgPyr[LEVEL].data;
        SE3Set[n].resize(6);
        cv::Mat se3_n = mFrameVecBuffer[n].GetDoubleSE3();
        std::memcpy(SE3Set[n].data(), (double*)se3_n.ptr<double>(0), 6*sizeof(double));

        for (int i = 0; i < mRefFrame.mKpsPyr[LEVEL].size(); i++) {
            double u0 = double(mRefFrame.mKpsPyr[LEVEL][i].pt.x);
            double v0 = double(mRefFrame.mKpsPyr[LEVEL][i].pt.y);

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

    std::vector<double> residuals;
    std::vector<double> gradients;
    double chi2 = 0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), NULL, &residuals, &gradients, NULL);

    #ifndef __ANDROID__
    std::cout << "Final report:\n" << summary.FullReport();
    #endif

    int N = 5;
    for (int i = 0; i < depthSet.size(); i++) {
        Log_info("depth: {},  {} {} {} {} {}", i, depthSet[i],
                       residuals[5*i], residuals[5*i+1], residuals[5*i+2], residuals[5*i+3], residuals[5*i+4]);
        mRefFrame.mDepthPyr[LEVEL][i] = float(depthSet[i]);
    }

    ShowDepth(mRefFrame);
    for (int i = 0; i < mFrameVecBuffer.size(); i++) {
        CheckReprojection(mRefFrame, mFrameVecBuffer[i]);
        cv::waitKey(-1);
    }

    return true;
}

void DirectVO::CheckReprojection(Frame& ref, Frame& f)
{
    std::vector<cv::Point2f> pts0;
    std::vector<cv::Point2f> pts1;
    float fx = mK->fx / pow(mRefFrame.mScaleFactor, LEVEL);
    float fy = mK->fy / pow(mRefFrame.mScaleFactor, LEVEL);
    float cx = mK->cx / pow(mRefFrame.mScaleFactor, LEVEL);
    float cy = mK->cy / pow(mRefFrame.mScaleFactor, LEVEL);

    for (int i = 0; i < ref.mKpsPyr[LEVEL].size(); i++) {
        pts0.push_back(ref.mKpsPyr[LEVEL][i].pt);
        float x, y, z;
        x = (ref.mKpsPyr[LEVEL][i].pt.x - cx) / fx * ref.mDepthPyr[LEVEL][i];
        y = (ref.mKpsPyr[LEVEL][i].pt.y - cy) / fy * ref.mDepthPyr[LEVEL][i];
        z = ref.mDepthPyr[LEVEL][i];

        float* R = f.mR.ptr<float>(0);
        float* t = f.mt.ptr<float>(0);

        float u1 = fx * (R[0]*x + R[1]*y + R[2]*z + t[0]) / (R[6]*x + R[7]*y + R[8]*z + t[2]) + cx;
        float v1 = fy * (R[3]*x + R[4]*y + R[5]*z + t[1]) / (R[6]*x + R[7]*y + R[8]*z + t[2]) + cy;
        pts1.push_back(cv::Point2f(u1, v1));
    }

    #ifndef __ANDROID__
    cv::Mat img4Show = cv::Mat::zeros(ref.mImgPyr[LEVEL].rows, ref.mImgPyr[LEVEL].cols*2, CV_8UC1);
    ref.mImgPyr[LEVEL].copyTo(img4Show(cv::Rect(0, 0, ref.mImgPyr[LEVEL].cols, ref.mImgPyr[LEVEL].rows)));
    f.mImgPyr[LEVEL].copyTo(img4Show(cv::Rect(ref.mImgPyr[LEVEL].cols, 0, ref.mImgPyr[LEVEL].cols, ref.mImgPyr[LEVEL].rows)));
    cv::cvtColor(img4Show, img4Show, CV_GRAY2BGR);
    for (int i = 0; i < pts0.size(); i++) {
        cv::circle(img4Show, pts0[i], 3, cv::Scalar(255, 0, 0), 1);
        cv::circle(img4Show, pts1[i]+cv::Point2f(ref.mImgPyr[LEVEL].cols,0), 3, cv::Scalar(255, 0, 255), 1);
        cv::line(img4Show, pts0[i], pts1[i]+cv::Point2f(ref.mImgPyr[LEVEL].cols,0), cv::Scalar(0, 255, 0), 1, CV_AA);
    }
    cv::imshow("img align", img4Show);
    #endif
}

void DirectVO::ShowDepth(Frame& f)
{
    cv::Mat imgDepth = cv::Mat::zeros(f.mImgPyr[LEVEL].rows, f.mImgPyr[LEVEL].cols, CV_32FC1);
    for (int i = 0; i < f.mDepthPyr[LEVEL].size(); i++) {
        // imgDepth.at<float>(int(f.mKpsPyr[LEVEL][i].pt.y), int(f.mKpsPyr[LEVEL][i].pt.x)) = f.mDepthPyr[LEVEL][i];
        // std::cout << f.mKpsPyr[LEVEL][i].pt << f.mDepthPyr[LEVEL][i] << " "
        //           << int(255.*f.mDepthPyr[LEVEL][i]) << " " << int(255.*(1.-f.mDepthPyr[LEVEL][i]))  
        //           << std::endl;
        imgDepth.at<float>(int(f.mKpsPyr[LEVEL][i].pt.y), int(f.mKpsPyr[LEVEL][i].pt.x)) 
                = f.mDepthPyr[LEVEL][i] / 20.0;
    }
    cv::imshow("depth", imgDepth);
}