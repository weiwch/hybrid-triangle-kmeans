/*
 * KmeansOrOrInitializer.cpp
 *
 *  Created on: Nov 30, 2016
 *      Author: wkwedlo
 */

#include <limits>
#include "../Util/OpenMP.h"
#include "../Util/Rand.h"
#include "KMeansOrOrInitializer.h"
#include "PlusPlusInitializer/PlusPlusInitializer.h"
#include "KMeansReportWriter.h"

KMeansOrOrInitializer::KMeansOrOrInitializer(NumaDataset &D,CentroidVector &aCV,int cl,double Os) : KMeansInitializer(D,aCV,cl),NumaData(D) {
	int nRows=NumaData.GetRowCount();
	ClosestDistances.SetSize(nRows);
	ClosestCentroids.SetSize(nRows);
	Oversample=Os;
	if (Os<=0.1)
		Rounds=10;
	else
		Rounds=5;

	int nThreads=omp_get_max_threads();
	ThreadData.SetSize(nThreads);
}

void KMeansOrOrInitializer::InitDataStructures() {
	int nRows=NumaData.GetRowCount();
#pragma omp parallel for default(none) firstprivate(nRows)
	for(int i=0;i<nRows;i++) {
		ClosestDistances[i]=std::numeric_limits<OPTFLOAT>::infinity();
		ClosestCentroids[i]=-1;
	}
}

void KMeansOrOrInitializer::InsertCentroid(DynamicArray<float> &cntr) {
	int nRows=NumaData.GetRowCount();
	int ThisCenter=Centers.GetSize();
#pragma omp parallel for default(none) firstprivate(nRows,ThisCenter) shared(cntr)
	for(int i=0;i<nRows;i++) {
		const DataRow &row=NumaData.GetRow(i);
		OPTFLOAT sqDist=CV.SquaredDistance(cntr,row);
		if (sqDist<ClosestDistances[i]) {
			ClosestDistances[i]=sqDist;
			ClosestCentroids[i]=ThisCenter;
		}
	}
	Centers.Add(cntr);
}

void KMeansOrOrInitializer::InsertCentroids() {
	int nRows=NumaData.GetRowCount();
	int ThisCenter=Centers.GetSize();
#pragma omp parallel for default(none) firstprivate(nRows,ThisCenter)
	for(int i=0;i<nRows;i++) {
		const DataRow &row=NumaData.GetRow(i);
		for(int j=0;j<NewCenters.GetSize();j++) {
			const DynamicArray<float> &cntr=NewCenters[j];
			OPTFLOAT sqDist=CV.SquaredDistance(cntr,row);
			if (sqDist<ClosestDistances[i]) {
				ClosestDistances[i]=sqDist;
				ClosestCentroids[i]=ThisCenter+j;
			}
		}
	}
	for(int i=0;i<NewCenters.GetSize();i++)
		Centers.Add(NewCenters[i]);
}


OPTFLOAT KMeansOrOrInitializer::ComputeCost() {
	OPTFLOAT ssq=0.0;
	int nRows=NumaData.GetRowCount();
#pragma omp parallel for default(none) firstprivate(nRows) reduction(+:ssq)
	for(int i=0;i<nRows;i++) {
		ssq+=ClosestDistances[i];
	}
	return ssq;
}

void KMeansOrOrInitializer::ScanDataset(OPTFLOAT Cost) {
	int nRows=NumaData.GetRowCount();
#pragma omp parallel default(none) firstprivate(nRows,Cost)
	{
		int tid=omp_get_thread_num();

		DynamicArray< DynamicArray<float> > &tNewCenters=ThreadData[tid].NewCenters;
		tNewCenters.SetSize(0);

#pragma omp for
		for(int i=0;i<nRows;i++) {
			OPTFLOAT Prob=Oversample*(OPTFLOAT)nclusters*ClosestDistances[i]/Cost;
			if (Rand()<Prob) {
				DataRow &row=NumaData.GetRow(i);
				tNewCenters.Add(row);
			}
		}
	}
}

void KMeansOrOrInitializer::MergeThreadCenters() {
	int nThreads=omp_get_max_threads();
	for(int i=0;i<nThreads;i++) {
		DynamicArray< DynamicArray<float> > &tNewCenters=ThreadData[i].NewCenters;
		for(int j=0;j<tNewCenters.GetSize();j++)
			NewCenters.Add(tNewCenters[j]);
	}
}

void KMeansOrOrInitializer::ComputeWeights() {
	int nCenters=Centers.GetSize();
	int nRows=NumaData.GetRowCount();

	Weights.SetSize(nCenters);

	for (int i=0;i<nCenters;i++)
		Weights[i]=0.0;

	for(int i=0;i<nRows;i++)
		Weights[ClosestCentroids[i]]+=(OPTFLOAT)1.0;

}

void KMeansOrOrInitializer::dbgDumpCentersAndWeights() {
#ifdef _DEBUG
	int nCenters=Centers.GetSize();
	OPTFLOAT WeightSum=(OPTFLOAT)0.0;
	for(int i=0;i<nCenters;i++) {
		const DataRow &Center=Centers[i];
		for (int j=0;j<ncols;j++)
			printf("%g ",Center[j]);
		printf(" : %g\n",Weights[i]);
		WeightSum+=Weights[i];
	}
	printf("Sum of all weights is %g\n",WeightSum);
#endif
}

void KMeansOrOrInitializer::Init(Array<OPTFLOAT> &v) {

	InitDataStructures();
	int FirstObj=Rand()*NumaData.GetTotalRowCount();
	DynamicArray<float> row(ncols);
	NumaData.GlobalFetchRow(FirstObj,row);
	InsertCentroid(row);

	OPTFLOAT Cost=ComputeCost();
	TRACE1("Initial cost is %g\n",Cost);

	for (int i=0;i<Rounds;i++) {
		NewCenters.SetSize(0);
		ScanDataset(Cost);
		MergeThreadCenters();
		InsertCentroids();
		Cost=ComputeCost();
		TRACE3("After round %d %d centers cost is %g\n",i,Centers.GetSize(),Cost);
	}
	ComputeWeights();
	//dbgDumpCentersAndWeights();

	StdDataset TempData(Centers);
	PlusPlusInitializer FinalInit(TempData,CV,nclusters);
	FinalInit.SetWeights(Weights);
	FinalInit.Init(v);

	//KMeansReportWriter Writer(NumaData,nclusters,NULL);
	//Writer.DumpClusters(v);
}
