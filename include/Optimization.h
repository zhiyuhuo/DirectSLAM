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

inline double GetPatchIntense(float u, float v, int width, unsigned char* image)
{
    double res = 0.0;
    for (int dv = -3; dv <= 4; dv++) {
        for (int du = -3; du <= 4; du++) {
            res += double(image[int(u)+du+(int(v)+dv)*width]);
        }
    }
    return res;
}

class PhotoMetricErrorDepthCostFuntion
: public ceres::SizedCostFunction<1, 1> {

    public:
    virtual ~PhotoMetricErrorDepthCostFuntion() {}

    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {

        bool ifNotUsePatch = true;
        double I0, I1;
        // First, compute the residual of intense.
        // 1. prepare data
        double Depth0  = parameters[0][0];
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

        // compute the intense of the pixel in the first image. 
        double xx0 = u0 - int(u0);
        double yy0 = v0 - int(v0);
        float WLT0 = (1-xx0)*(1-yy0);
        float WRT0 = xx0*(1-yy0);
        float WLB0 = (1-xx0)*yy0;
        float WRB0 = xx0*yy0;

        if (ifNotUsePatch) {
            I0  =   image0[int(u0)   + width*int(v0)]   * WLT0 
                        + image0[int(u0)+1 + width*int(v0)]   * WRT0
                        + image0[int(u0)   + width*(int(v0)+1)] * WLB0
                        + image0[int(u0)+1 + width*(int(v0)+1)] * WRB0;
        } else {
            I0  = GetPatchIntense(u0, v0, width, image0) * WLT0
                        + GetPatchIntense(u0+1, v0, width, image0) * WRT0
                        + GetPatchIntense(u0, v0+1, width, image0) * WLB0
                        + GetPatchIntense(u0+1, v0+1, width, image0) * WRB0;
        }

        // compute the intense of the pixel in the second image.
        double u1 = x1 * fx / z1 + cx;
        double v1 = y1 * fy / z1 + cy;

        double xx1 = u1 - int(u1);
        double yy1 = v1 - int(v1);
        float WLT1 = (1-xx1)*(1-yy1);
        float WRT1 = xx1*(1-yy1);
        float WLB1 = (1-xx1)*yy1;
        float WRB1 = xx1*yy1;

        if (ifNotUsePatch) {
            I1  =   image1[int(u1)   + width*int(v1)]   * WLT1 
                        + image1[int(u1)+1 + width*int(v1)]   * WRT1
                        + image1[int(u1)   + width*int((v1)+1)] * WLB1
                        + image1[int(u1)+1 + width*int((v1)+1)] * WRB1;
        } else {
            I1  = GetPatchIntense(u1, v1, width, image1) * WLT1
                        + GetPatchIntense(u1+1, v1, width, image1) * WRT1
                        + GetPatchIntense(u1, v1+1, width, image1) * WLB1
                        + GetPatchIntense(u1+1, v1+1, width, image1) * WRB1;
        }

        // if (track == 50) {
        //     std::cout << SE3[0] << " " << SE3[1] << " " << SE3[2] << " " << SE3[3] << " " << SE3[4] << " " << SE3[5] << std::endl;
        //     std::cout << Depth0 << std::endl;
        //     std::cout << p0[0] << " " << p0[1] << " " << p0[2] << std::endl;
        //     std::cout << x1 << " " << y1 << " " << z1 << std::endl;
        //     std::cout << u0 << " " << v0 << " " << u1 << " " << v1 << std::endl;
        //     std::cout << I0 << " " << I1 << std::endl;
        // }

        residuals[0] = I1 - I0;

        if (u1 < 1 || u1 >= width-1 || v1 < 1 || v1 >= height-1) {
            residuals[0] = 0.;
        }

        // Second, compute the jacobian.
        double z1_2 = z1*z1;
        double inv_z1 = 1 / z1;
        double inv_z1_2 = 1 / z1_2;
        if (jacobians != NULL && jacobians[0] != NULL) {

            int u0L = int(u0)-1;
            int v0L = int(v0);
            int u0R = int(u0)+1;
            int v0R = int(v0);
            int u0T = int(u0);
            int v0T = int(v0)-1;
            int u0B = int(u0);
            int v0B = int(v0)+1;

            double I0L, I0R, I0T, I0B;
            if (ifNotUsePatch) {
                I0L  =   image0[u0L   + width*v0L]   * WLT0 
                            + image0[u0L+1 + width*v0L]   * WRT0
                            + image0[u0L   + width*(v0L+1)] * WLB0
                            + image0[u0L+1 + width*(v0L+1)] * WRB0;
                I0R  =   image0[u0R   + width*v0R]   * WLT0 
                            + image0[u0R+1 + width*v0R]   * WRT0
                            + image0[u0R   + width*(v0R+1)] * WLB0
                            + image0[u0R+1 + width*(v0R+1)] * WRB0;
                I0T  =   image0[u0T   + width*v0T]   * WLT0 
                            + image0[u0T+1 + width*v0T]   * WRT0
                            + image0[u0T   + width*(v0T+1)] * WLB0
                            + image0[u0T+1 + width*(v0T+1)] * WRB0;
                I0B  =   image0[u0B   + width*v0B]   * WLT0 
                            + image0[u0B+1 + width*v0B]   * WRT0
                            + image0[u0B   + width*(v0B+1)] * WLB0
                            + image0[u0B+1 + width*(v0B+1)] * WRB0;
            } else {
                I0L  = GetPatchIntense(u0L, v0L, width, image0) * WLT0
                                + GetPatchIntense(u0L+1, v0L, width, image0) * WRT0
                                + GetPatchIntense(u0L, v0L+1, width, image0) * WLB0
                                + GetPatchIntense(u0L+1, v0L+1, width, image0) * WRB0;
                I0R  = GetPatchIntense(u0R, v0R, width, image0) * WLT0
                                + GetPatchIntense(u0R+1, v0R, width, image0) * WRT0
                                + GetPatchIntense(u0R, v0R+1, width, image0) * WLB0
                                + GetPatchIntense(u0R+1, v0R+1, width, image0) * WRB0;
                I0T  = GetPatchIntense(u0T, v0T, width, image0) * WLT0
                                + GetPatchIntense(u0T+1, v0T, width, image0) * WRT0
                                + GetPatchIntense(u0T, v0T+1, width, image0) * WLB0
                                + GetPatchIntense(u0T+1, v0T+1, width, image0) * WRB0;
                I0B  = GetPatchIntense(u0B, v0B, width, image0) * WLT0
                                + GetPatchIntense(u0B+1, v0B, width, image0) * WRT0
                                + GetPatchIntense(u0B, v0B+1, width, image0) * WLB0
                                + GetPatchIntense(u0B+1, v0B+1, width, image0) * WRB0;
            }

            int u1L = int(u1)-1;
            int v1L = int(v1);
            int u1R = int(u1)+1;
            int v1R = int(v1);
            int u1T = int(u1);
            int v1T = int(v1)-1;
            int u1B = int(u1);
            int v1B = int(v1)+1;

            double I1L, I1R, I1T, I1B;
            if (ifNotUsePatch) {
                I1L  =   image1[u1L   + width*v1L]   * WLT1 
                            + image1[u1L+1 + width*v1L]   * WRT1
                            + image1[u1L   + width*(v1L+1)] * WLB1
                            + image1[u1L+1 + width*(v1L+1)] * WRB1;
                I1R  =   image1[u1R   + width*v1R]   * WLT1 
                            + image1[u1R+1 + width*v1R]   * WRT1
                            + image1[u1R   + width*(v1R+1)] * WLB1
                            + image1[u1R+1 + width*(v1R+1)] * WRB1;
                I1T  =   image1[u1T   + width*v1T]   * WLT1 
                            + image1[u1T+1 + width*v1T]   * WRT1
                            + image1[u1T   + width*(v1T+1)] * WLB1
                            + image1[u1T+1 + width*(v1T+1)] * WRB1;
                I1B  =   image1[u1B   + width*v1B]   * WLT1 
                            + image1[u1B+1 + width*v1B]   * WRT1
                            + image1[u1B   + width*(v1B+1)] * WLB1
                            + image1[u1B+1 + width*(v1B+1)] * WRB1;
            } else {
                I1L  = GetPatchIntense(u1L, v1L, width, image1) * WLT1
                            + GetPatchIntense(u1L+1, v1L, width, image1) * WRT1
                            + GetPatchIntense(u1L, v1L+1, width, image1) * WLB1
                            + GetPatchIntense(u1L+1, v1L+1, width, image1) * WRB1;
                I1R  = GetPatchIntense(u1R, v1R, width, image1) * WLT1
                            + GetPatchIntense(u1R+1, v1R, width, image1) * WRT1
                            + GetPatchIntense(u1R, v1R+1, width, image1) * WLB1
                            + GetPatchIntense(u1R+1, v1R+1, width, image1) * WRB1;
                I1T  = GetPatchIntense(u1T, v1T, width, image1) * WLT1
                            + GetPatchIntense(u1T+1, v1T, width, image1) * WRT1
                            + GetPatchIntense(u1T, v1T+1, width, image1) * WLB1
                            + GetPatchIntense(u1T+1, v1T+1, width, image1) * WRB1;
                I1B  = GetPatchIntense(u1B, v1B, width, image1) * WLT1
                            + GetPatchIntense(u1B+1, v1B, width, image1) * WRT1
                            + GetPatchIntense(u1B, v1B+1, width, image1) * WLB1
                            + GetPatchIntense(u1B+1, v1B+1, width, image1) * WRB1;
            }

            double d_I0_u = (I0R - I0L) * 0.5;
            double d_I0_v = (I0B - I0T) * 0.5;
            double d_I1_u = (I1R - I1L) * 0.5;
            double d_I1_v = (I1B - I1T) * 0.5;

            double d_I_u = d_I1_u;
            double d_I_v = d_I1_v;

            // 3. Get the du/dDepth
            double R[9] = {};
            ceres::AngleAxisToRotationMatrix(SE3, R);
            double d_u_Depth  = fx * ( x1*(-R[8]*inv_z1_2) + inv_z1*R[6]);
            double d_v_Depth  = fy * ( y1*(-R[8]*inv_z1_2) + inv_z1*R[7]);

            // 4. compute the jacobian of dI/d(para)
            jacobians[0][0] = d_I_u * d_u_Depth + d_I_v * d_v_Depth;

            // std::cout << jacobians[0][0] << std::endl;
        }

        //  std::cout << "........................................" << std::endl;
        return true;
    }

    PhotoMetricErrorDepthCostFuntion():
    u0(0), v0(0), width(0), height(0), fx(0), fy(0), cx(0), cy(0), image0(NULL), image1(NULL), SE3(NULL), track(0)
    {
    }

    PhotoMetricErrorDepthCostFuntion(const double u0_, const double v0_, const int width_, const int height_,
                                     const double fx_, const double fy_, const double cx_, const double cy_,
                                     unsigned char* image0_, unsigned char* image1_, double* SE3_,
                                     int track_):
        u0(u0_), v0(v0_), width(width_), height(height_), 
        fx(fx_), fy(fy_), cx(cx_), cy(cy_), 
        image0(image0_), image1(image1_),
        SE3(SE3_),
        track(track_)
    {
    }

    /* the pixel in the first frame */
    const double    u0;
    const double    v0;
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
    /* the transform of the first frame and the second frame */
    double* SE3;

    /*problem num. for tracking*/
    const int    track;

};


}

#endif;