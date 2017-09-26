#ifndef DIRECTSLAM_OPTIMIZATION_H_
#define DIRECTSLAM_OPTIMIZATION_H_

#include <cstdio>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <vector>
#include <stdint.h>
#include <unistd.h>


#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "Log.h"
#include "Timer.h"

namespace OptimizationCeres {

class PhotoMetricErrorSE3DepthCostFuntion
: public ceres::SizedCostFunction<1, 6, 1> {

    public:
    virtual ~PhotoMetricErrorSE3DepthCostFuntion() {}

    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {

        // First, compute the residual of intense.
        // 1. prepare data
        double SE3[6] = { parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5] };
        double Depth0  = parameters[1][0];
        double p0[3] = { (u0-cx)/fx*Depth0, (v0-cy)/fy*Depth0, Depth0 };
        double p1[3] = {};
        ceres::AngleAxisRotatePoint(SE3, p0, p1);
        p1[0] += SE3[3];
        p1[1] += SE3[4];
        p1[2] += SE3[5];

        // 2. compute the residual

        double x1 = p1[0];
        double y1 = p1[1];
        double z1 = p1[2];

        int u1 = int(x1 * fx / z1 + cx + 0.5);
        int v1 = int(y1 * fy / z1 + cy + 0.5);

        double I0 = double(image0[u0 + width*v0]);
        double I1 = double(image1[u1 + width*v1]);

        std::cout << SE3[0] << " " << SE3[1] << " " << SE3[2] << " " << SE3[3] << " " << SE3[4] << " " << SE3[5] << std::endl;
        std::cout << Depth0 << std::endl;
        std::cout << p0[0] << " " << p0[1] << " " << p0[2] << std::endl;
        std::cout << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
        std::cout << x1 << " " << y1 << " " << z1 << std::endl;
        std::cout << u0 << " " << v0 << " " << u1 << " " << v1 << std::endl;
        std::cout << I0 << " " << I1 << std::endl;
        std::cout << track << std::endl;

        residuals[0] = I0 - I1;

        // Second, compute the jacobian.
        double z1_2 = z1*z1;
        double inv_z1 = 1 / z1;
        double inv_z1_2 = 1 / z1_2;
        if (jacobians != NULL && jacobians[0] != NULL) {
            std::cout << "compute the jacobian: " << std::endl;

            // 1. Get dI/du (use the first frame since we are not sure we can track gradient in the second frame)
            double d_I_u = double(image1[u0 + 1 + width*v0]) - double(image1[u0 - 1 + width*v0]);
            double d_I_v = double(image1[u0 + width*(v0+1)]) - double(image1[u0 + width*(v0-1)]);

            // 2. Get du/dq*dq/dlie = du/dlie Jacobians of SE3
            double d_u_rx = -fx*x1*y1*inv_z1_2;
            double d_u_ry = fx + fx*x1*x1*inv_z1_2;
            double d_u_rz = -fx*y1*inv_z1;

            double d_v_rx = -fy-fy*y1*y1*inv_z1_2;
            double d_v_ry = fy*x1*y1*inv_z1_2;
            double d_v_rz = fy*x1*inv_z1;

            double d_u_tx = fx*inv_z1;
            double d_u_ty = 0;
            double d_u_tz = -fx*x1*inv_z1_2;

            double d_v_tx = 0;
            double d_v_ty = fy*inv_z1;
            double d_v_tz = -fy*y1*inv_z1_2;  

            // 3. Get the du/dDepth
            double R[9] = {};
            ceres::AngleAxisToRotationMatrix(SE3, R);
            double d_u_Depth  = fx * ( x1*(-R[8]*inv_z1_2) + inv_z1*R[6]);
            double d_v_Depth  = fy * ( y1*(-R[8]*inv_z1_2) + inv_z1*R[7]);

            // 4. compute the jacobian of dI/d(para)
            jacobians[0][0] = d_I_u * d_u_rx + d_I_v * d_v_rx;
            jacobians[0][1] = d_I_u * d_u_ry + d_I_v * d_v_ry;
            jacobians[0][2] = d_I_u * d_u_rz + d_I_v * d_v_rz;
            jacobians[0][3] = d_I_u * d_u_tx + d_I_v * d_v_tx;
            jacobians[0][4] = d_I_u * d_u_ty + d_I_v * d_v_ty;
            jacobians[0][5] = d_I_u * d_u_tz + d_I_v * d_v_tz;
            jacobians[1][0] = d_I_u * d_u_Depth + d_I_v * d_v_Depth;
 
            std::cout << "compute jacobian" << std::endl;
            std::cout << jacobians[0][0] << " " << jacobians[0][1] << " " << jacobians[0][2] << " " 
                      << jacobians[0][3] << " " << jacobians[0][4] << " " << jacobians[0][5] << std::endl;

            std::cout << jacobians[1][0] << std::endl;
        }

        std::cout << "........................................" << std::endl;
        return true;
    }

    PhotoMetricErrorSE3DepthCostFuntion():
    u0(0), v0(0), width(0), height(0), fx(0), fy(0), cx(0), cy(0), image0(NULL), image1(NULL), track(0)
    {
    }

    PhotoMetricErrorSE3DepthCostFuntion(const int u0_, const int v0_, const int width_, const int height_,
                                        const double fx_, const double fy_, const double cx_, const double cy_,
                                        unsigned char* image0_, unsigned char* image1_,
                                        int track_):
        u0(u0_), v0(v0_), width(width_), height(height_), 
        fx(fx_), fy(fy_), cx(cx_), cy(cy_), 
        image0(image0_), image1(image1_),
        track(track_)
    {
    }

    /* the pixel in the first frame */
    const int    u0;
    const int    v0;
    const int    width;
    const int    height; 

    /* the camera intrinsic */
    const double fx;
    const double fy;
    const double cx;
    const double cy;

    /* the first image and the second image */
    unsigned char* image0;
    unsigned char* image1;

    /*problem num. for tracking*/
    const int    track;
};


}

#endif;