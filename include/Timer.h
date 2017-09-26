#ifndef DIRECTSLAM_TIMER_H
#define DIRECTSLAM_TIMER_H

#include <sys/time.h>
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <chrono>
#include "Log.h"

#ifdef NO_TIMING
#define TIME_BEGIN() ;
#define TIME_END(TAG) ;
#else
#define TIME_BEGIN() { struct timeval t_b, t_e; \
    gettimeofday(&t_b, NULL);
#define TIME_END(TAG) gettimeofday(&t_e, NULL); \
    double time_used = (t_e.tv_sec - t_b.tv_sec) + (t_e.tv_usec - t_b.tv_usec) * 1e-6; \
    Log_info("%% {} TIME:{}s", TAG, time_used); }
#endif

#endif
