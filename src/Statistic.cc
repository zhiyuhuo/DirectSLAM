#include "Statistic.h"

void Statistic::clear()
{
    mu = 0;
    sigma2 = 0;
    n = 0;
}

void Statistic::add(float d)
{
    float mu_t = (n*mu + d) / (n + 1);
    sigma2 = ((sigma2 + mu*mu)*n + d*d) / (n + 1) - mu_t*mu_t; 

    mu = mu_t;
    ++n;
}

float Statistic::getMean()
{
    if (!n)
        return -1;

    return mu;
}

float Statistic::getVariance()
{
    if (!n)
        return -1;

    return sigma2;
}

float Statistic::getStdVariance()
{
    if (!n)
        return -1;
        
    return sqrt(sigma2);
}

float Statistic::getNum()
{
    return float(n);
}


