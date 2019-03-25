#include <math.h>
#include "Rand.h"
#include "MersenneTwister.h"
#include <unistd.h>

drand48_data Buffer1;


MTRand *pRNG;

#ifdef _OPENMP
#pragma omp threadprivate(pRNG)
#endif


double Rand() 
{
	return pRNG->rand53();
}


double RandCauchy()
{
	return tan(M_PI*(pRNG->rand53()-0.5));
}

double RandNorm()
{
	return pRNG->randNorm(0.0,1.0);
}

int RandInt() {
	return pRNG->randInt();
}

int RandInt(const unsigned long n) {
	return pRNG->randInt(n);
}

double RandDbl() {
	return pRNG->randDblExc();
}

double RandDbl(const double n) {
	return pRNG->randDblExc(n);
}

void SRand(unsigned seed) 
{
	if (pRNG==NULL)
		pRNG=new MTRand(123456);
	pRNG->seed(seed*7);
}

void DelMTRand() {
	if (pRNG!=NULL)
		delete pRNG;
}

void SRand1(unsigned seed) 
{
	srand48_r((long)seed*1111,&Buffer1);
}
/*
int SaveRandState(FILE *f) {
	MTRand::uint32 buff[RNG.SAVE];
	RNG.save(buff);
//	printf("Saved state:\n");
//	for (int i=0; i<RNG.SAVE; i++)
//		printf("%d ", buff[i]);ls
//	printf("\n");
//	return fwrite(buff, sizeof(MTRand::uint32), RNG.SAVE, f) - RNG.SAVE;	
	return write(fileno(f), buff, sizeof(MTRand::uint32) * RNG.SAVE) - sizeof(MTRand::uint32)*RNG.SAVE;
}

int LoadRandState(FILE *f) {
	MTRand::uint32 buff[RNG.SAVE];
	int ret = fread(buff, sizeof(MTRand::uint32), RNG.SAVE, f) - RNG.SAVE;
	if (ret < 0) 
		return ret;
//	printf("Loaded state:\n");
//	for (int i=0; i<RNG.SAVE; i++)
//		printf("%d ", buff[i]);
//	printf("\n");

	RNG.load(buff);	
	return ret;
}

int GetRandStateSize() {
	return RNG.SAVE*sizeof(MTRand::uint32);
}

int GetRandState(char* buff) {
	RNG.save((MTRand::uint32*) buff);
	return 0;
}

int SetRandState(char* buff) {
	RNG.load((MTRand::uint32*) buff);
	return 0;
}*/
