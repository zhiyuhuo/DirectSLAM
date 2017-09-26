#include "DirectVO.h" 

DirectVO::DirectVO(CameraIntrinsic* K)
{
    mK = K;
    mState = TrackingState::VOID;
}

bool DirectVO::SetFirstFrame(Frame& f)
{
    f.ExtractFASTPyr();
    f.InitDepthPyr(10.0);
    f.ShowPyr();

    mRefFrame = f;
    return true;
}

TrackingState DirectVO::TrackMono(cv::Mat image) 
{
    Frame f(image);

    if (mState == TrackingState::VOID) {
        SetFirstFrame(f);
        mState = TrackingState::INITIALIZING;
    }
    else if (mState == TrackingState::INITIALIZING) {
        TrackRefFrame(f);
    }

    return mState;
}

bool DirectVO::TrackRefFrame(Frame& f)
{
    // Define the problem instance
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // track the points in the highest level: 3
    int LEVEL = 3;
    float fx = mK->fx / pow(mRefFrame.mScaleFactor, LEVEL);
    float fy = mK->fy / pow(mRefFrame.mScaleFactor, LEVEL);
    float cx = mK->cx / pow(mRefFrame.mScaleFactor, LEVEL);
    float cy = mK->cy / pow(mRefFrame.mScaleFactor, LEVEL);
    int width = mK->width / pow(mRefFrame.mScaleFactor, LEVEL);
    int height = mK->height / pow(mRefFrame.mScaleFactor, LEVEL);
    unsigned char* image0 = mRefFrame.mImgPyr[LEVEL].data;
    unsigned char* image1 = f.mImgPyr[LEVEL].data;

    // set the parameters to optimize: Camera Transformation and Depths
    double current_camera_R_t[6] = {0, 0, 0, 0, 0, 0};
    std::vector<double> depthSet(mRefFrame.mDepthPyr[LEVEL].size());
    for (int i = 0; i < mRefFrame.mDepthPyr[LEVEL].size(); i++) {
        depthSet[i] = double(mRefFrame.mDepthPyr[LEVEL][i]);
    }

    for (int i = 0; i < mRefFrame.mKpsPyr[LEVEL].size(); i++) {
        int u0 = int(mRefFrame.mKpsPyr[LEVEL][i].pt.x+0.5);
        int v0 = int(mRefFrame.mKpsPyr[LEVEL][i].pt.y+0.5);

        problem.AddResidualBlock(new OptimizationCeres::PhotoMetricErrorSE3DepthCostFuntion(
            u0, v0, width, height, fx, fy, cx, cy, image0, image1, i),
            NULL, current_camera_R_t, &depthSet[i]);
    }

    // To the optimization
    Log_info("Start to solve the problem...");
    // Configure the solver. // for bundle adjustment
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;

    options.num_threads = 1;
    options.num_linear_solver_threads = 1;
    options.use_nonmonotonic_steps = false;

    options.preconditioner_type = ceres::JACOBI;
    // options.preconditioner_type = ceres::SCHUR_JACOBI;

    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.linear_solver_type = ceres::DENSE_QR
    ;
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
    double chi2 = 0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), NULL, &residuals, NULL, NULL);
    for (int i = 0; i < residuals.size(); i++) {
        Log_info("residuals {}: {}", i, residuals[i]);
    }

    #ifndef __ANDROID__
    std::cout << "Final report:\n" << summary.FullReport();
    #endif

    std::cout << current_camera_R_t[0] << " " 
                << current_camera_R_t[1] << " " 
                << current_camera_R_t[2] << " " 
                << current_camera_R_t[3] << " " 
                << current_camera_R_t[4] << " " 
                << current_camera_R_t[5] << std::endl;

    for (int i = 0; i < depthSet.size(); i++) {
        Log_info("updated depth {}: {}", i, depthSet[i]);
    }

    return false;
}