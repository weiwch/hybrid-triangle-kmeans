/*
 * OpenMPKMAReducer.cpp
 *
 *  Created on: Feb 10, 2016
 *      Author: wkwedlo
 */

#include <unistd.h>

#include "OpenMPKMAReducer.h"
#include "../Util/OpenMP.h"

OpenMPKMAReducer::OpenMPKMAReducer(CentroidVector &CV) {
	NThreads=omp_get_max_threads();
	nclusters=CV.GetNClusters();
	ncols=CV.GetNCols();
	stride=ncols;

	ThreadData.SetSize(NThreads);
	int CenterSize=nclusters*ncols;

	#pragma omp parallel default(none)
	{
		int i=omp_get_thread_num();
		ThreadData[i].Center.SetSize(nclusters*ncols);
		ThreadData[i].Counts.SetSize(nclusters);
		ClearThreadData(i);
	}
}

void OpenMPKMAReducer::ClearThreadData(int Thread) {
	ThreadPrivateVector<int> &Counts=GetThreadCounts(Thread);

#pragma omp simd
	for(int i=0;i<nclusters;i++)
		Counts[i]=0;

	ThreadPrivateVector<OPTFLOAT> &Center=GetThreadCenter(Thread);
	int CenterSize=Center.GetSize();

#pragma omp simd
	for(int i=0;i<CenterSize;i++)
		Center[i]=(OPTFLOAT)0.0;
}

void OpenMPKMAReducer::ClearArrays(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts) {
	int CountSize=Counts.GetSize();

#pragma omp simd
	for(int i=0;i<CountSize;i++)
		Counts[i]=0;
	int CenterSize=Center.GetSize();

#pragma omp simd
	for(int i=0;i<CenterSize;i++)
		Center[i]=(OPTFLOAT)0.0;
}

void OpenMPKMAReducer::AddZeroToCenter(ThreadPrivateVector<OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts) {
	ThreadPrivateVector<OPTFLOAT> &zeroCenter = ThreadData[0].Center;
	ThreadPrivateVector<int> &zeroCounts = ThreadData[0].Counts;


	int CenterSize = nclusters * ncols;
#pragma omp simd
	for(int i = 0; i < CenterSize; i++) {
		Center[i] += zeroCenter[i];
	}

#pragma omp simd
	for(int i = 0; i < nclusters; i++) {
		Counts[i] += zeroCounts[i];
	}

}



NaiveOpenMPReducer::NaiveOpenMPReducer(CentroidVector &CV) : OpenMPKMAReducer(CV) {

}

void NaiveOpenMPReducer::ReduceToZero() {
	int CenterSize = nclusters * ncols;
	ThreadPrivateVector<OPTFLOAT> &zeroCenter = ThreadData[0].Center;
	ThreadPrivateVector<int> &zeroCounts = ThreadData[0].Counts;
	OPTFLOAT *  __restrict__  pZC=zeroCenter.GetData();

	for (int i = 1; i < NThreads; i++) {
		const ThreadPrivateVector<OPTFLOAT> &myCenter = ThreadData[i].Center;
		const ThreadPrivateVector<int> &myCounts = ThreadData[i].Counts;
		const OPTFLOAT * __restrict__  pMC=myCenter.GetData();

#pragma omp simd
		for (int j = 0; j < CenterSize; j++) {
			pZC[j] += pMC[j];
		}

#pragma omp simd
		for (int j = 0; j < nclusters; j++) {
			zeroCounts[j] += myCounts[j];
		}
	}
}

void NaiveOpenMPReducer::ReduceToArrays(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts) {
	const int CenterSize = nclusters * ncols;

	OPTFLOAT *  __restrict__  pC=Center.GetData();

	for (int i = 0; i < NThreads; i++) {
		const ThreadPrivateVector<OPTFLOAT> &myCenter = ThreadData[i].Center;
		const ThreadPrivateVector<int> &myCounts = ThreadData[i].Counts;
		const OPTFLOAT * __restrict__  pMC=myCenter.GetData();

#pragma omp simd
		for (int j = 0; j < CenterSize; j++) {
			pC[j] += pMC[j];
		}
#pragma omp simd
		for (int j = 0; j < nclusters; j++) {
			Counts[j] += myCounts[j];
		}
	}
}

// Tree-based Reduction
Log2OpenMPReducer::Log2OpenMPReducer(CentroidVector &CV) : OpenMPKMAReducer(CV) {

}

//#define _TRACE_REDUCE


void Log2OpenMPReducer::AddThreadArrays(int DstThr,int SrcThr) {
#ifdef _TRACE_REDUCE
	TRACE2("Log2OpenMPReducer: About to add thread %d to thread %d\n",SrcThr,DstThr);
#endif
	const int CenterSize = nclusters * ncols;



	OPTFLOAT * __restrict__ dstCenter = ThreadData[DstThr].Center.GetData();
	int * __restrict__ dstCounts = ThreadData[DstThr].Counts.GetData();

	const OPTFLOAT * __restrict__ srcCenter = ThreadData[SrcThr].Center.GetData();
	const int * __restrict__ srcCounts = ThreadData[SrcThr].Counts.GetData();


#pragma omp simd aligned(dstCenter: 4 *sizeof(OPTFLOAT)) aligned(srcCenter: 4 *sizeof(OPTFLOAT))
	for (int j = 0; j < CenterSize; j++) {
		dstCenter[j] += srcCenter[j];
	}

#pragma omp simd aligned(dstCounts: 8 *sizeof(int)) aligned(srcCounts: 8 *sizeof(int))
	for (int j = 0; j < nclusters; j++) {
		dstCounts[j] += srcCounts[j];
	}


}

void Log2OpenMPReducer::ReduceToZero() {

#pragma omp parallel
	{
		int tid=omp_get_thread_num();
		for(int s=1; s<NThreads;s*=2) {
			if (tid % (2*s)==0 && tid+s<NThreads)
				AddThreadArrays(tid,tid+s);
#pragma omp barrier
#ifdef _TRACE_REDUCE
#pragma omp single
			{
				TRACE0("Next round\n");
			}
#endif
		}
	}
}

void Log2OpenMPReducer::ReduceToArrays(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts) {
	ReduceToZero();
	memcpy(Center.GetData(),ThreadData[0].Center.GetData(),sizeof(OPTFLOAT)*ncols*nclusters);
	memcpy(Counts.GetData(),ThreadData[0].Counts.GetData(),sizeof(int)*nclusters);

}

