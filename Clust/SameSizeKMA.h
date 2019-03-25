/*
 * SameSizeKMA.h
 *
 *  Created on: Nov 8, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_SAMESIZEKMA_H_
#define CLUST_SAMESIZEKMA_H_

#include <queue>

#include "KMAlgorithm.h"
#include "CentroidRepair.h"
#include "../Util/StdDataset.h"

struct PointDistance {
	int Point;
	OPTFLOAT Distance;
	int Centroid;
	int operator<(const PointDistance &Other) const {return Distance < Other.Distance;}
};



class SameSizeKMA {

protected:
	CentroidVector &CV;
	KMeansAlgorithm *pKMA;
	int nclusters,ncols;
	CentroidRepair *pRepair;
	StdDataset &Data;
	std::priority_queue<PointDistance> heap;

	OPTFLOAT FindAssignment(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment);
	int FindTransfers(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment);

public:
	SameSizeKMA(CentroidVector &aCV,StdDataset &Data);
	~SameSizeKMA();
	OPTFLOAT Train(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment);
	OPTFLOAT TrainIterative(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment,int Iters,int Verbosity);
	OPTFLOAT TrainIterativeFromAssignment(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment,int Iters,int Verbosity);

	void DumpClusters(Array<OPTFLOAT> &vec,DynamicArray<int> Assignment);
};

#endif /* CLUST_SAMESIZEKMA_H_ */
