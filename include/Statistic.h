#ifndef TINYSLAM_STATISTIC_H_
#define TINYSLAM_STATISTIC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <vector>

using namespace std;

class Statistic
{
    public:
        Statistic():mu(0), sigma2(0), n(0) {};

        void  add(float d);
        void  clear();

        float getMean();
        float getVariance();
        float getStdVariance();
        float getNum();

    private:
        float mu;
        float sigma2;

        int   n;
};



#endif