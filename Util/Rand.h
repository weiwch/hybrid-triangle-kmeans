

#ifndef RAND_H
#define RAND_H
#include <cstdlib>
#include <cstdio>

extern drand48_data Buffer1;

double Rand();
double RandNorm();
double RandCauchy();

int RandInt();
int RandInt(const unsigned long n);

double RandDbl();
double RandDbl(const double n);

static inline double Rand1() {
    double result;
    drand48_r(&Buffer1,&result);
    return result;
}

int LoadRandState(FILE *f);
int SaveRandState(FILE *f);
int GetRandStateSize();
int GetRandState(char* buff);
int SetRandState(char* buff);
void SRand(unsigned seed);
void DelMTRand();

void SRand1(unsigned seed);
#endif

