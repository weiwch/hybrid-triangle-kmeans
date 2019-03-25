/*
 * YinyangKMA.h
 *
 *  Created on: Nov 12, 2015
 *      Author: wkwedlo
 */

#ifndef CLUST_YINYANGKMA_H_
#define CLUST_YINYANGKMA_H_

#include "KMAlgorithm.h"
#include "CentroidVectorPermutation.h"
#include "YinyangClusterer.h"
#include "../Util/LargeVector.h"
#include "../Util/LargeMatrix.h"

struct YinyangThreadData {

	/// Number of the closest center in each group
	DynamicArray<int> GroupBestIndex;


	/// Old lower bounds used in the local filtering stage
	DynamicArray<OPTFLOAT> TempLowerBounds;
	/// if true then i-th Group must be checked in local filtering stage
	DynamicArray<bool> GroupMask;

	ThreadPrivateVector<OPTFLOAT> row;
}__attribute__ ((aligned (64)));





class YinyangKMABase : public KMeansAlgorithm, public KMeansWithoutMSE {

protected:

	/// Number of centroid groups. 1<=t<=nclusters
	int t;
	bool InitialCluster;




	int ICntr;
	YinyangClusterer *pClust;
	CentroidVectorPermutation Perm;

	OpenMPKMAReducer *pOMPReducer;
	DynamicArray<YinyangThreadData> OMPData;
	LargeMatrix<OPTFLOAT> LowerBounds;
	LargeVector<OPTFLOAT> UpperBounds;
	/// Assignment in this iteration
	LargeVector<int> Assignment;

	ThreadPrivateVector<OPTFLOAT> Center;
	ThreadPrivateVector<int> Counts;

	/// Number of group of a cluster j
	DynamicArray<int> GroupNumbers;

	/// Number of the  first centroid in group of centroids
	DynamicArray<int> GroupFirst;

	/// Number of the  first centroid **not*** in group of centroids
	DynamicArray<int> GroupNotLast;

	/// Maximal movement in a group of centroids
	DynamicArray<OPTFLOAT> GroupMaxMoved;

	/// Centroid coordinates from the previous iteration
	DynamicArray<OPTFLOAT> tmp_vec;

	/// Distance each centroid moved
	DynamicArray<OPTFLOAT> DistanceMoved;


	void ComputeGroupSizes();
	void ComputeGroupNumbers();
	void InitCentroids(const Array<OPTFLOAT> &vec);
	void ComputeNewVec(Array<OPTFLOAT> &vec);
	void ComputeGroupDrifts();
	void MoveObject(const int PrevA, int NewA, const ThreadPrivateVector<OPTFLOAT> &row,int tid);
	void ClusterCenters(Array<OPTFLOAT> &vec,CentroidVectorPermutation &aPerm);
	void ClusterInitialCenters(Array<OPTFLOAT> &vec);
	void ClusterCentersUsingGroupNumbers(Array<OPTFLOAT> &vec,CentroidVectorPermutation &aPerm);


	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &SSE) {}
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bcont) {}

	void FirstIteration(const Array<OPTFLOAT> &vec);
	int ComputeAssignmentInFirstIter(const int i,const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row, long *DistanceCount);

	virtual bool OuterLoop(const Array<OPTFLOAT> &vec, long *DistanceCount);
	virtual void InitDataStructures(Array<OPTFLOAT> &vec);

	/// Debug function to verify whether Assignment[i] points to the closest cluster
	void DbgVerifyAssignment(int i,const Array<OPTFLOAT> &vec,const float * __restrict__ row);
	void DbgVerifyLowerBounds(int i,int bestj,const Array<OPTFLOAT> &vec,const float * __restrict__ row);
	void DbgPrintAll(const Array<OPTFLOAT> &vec,int FilterRow=-1);
	virtual void DbgPrintInfo(int i,const Array<OPTFLOAT> &vec,const float * __restrict__ row)=0;

	virtual void ReclusterCenters(Array<OPTFLOAT> &vec) {}

	int ComputeAssignmentWithFilterICML15(const int i,const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row, long *DistanceCount,
			const DynamicArray<bool> &GroupMask, const DynamicArray<OPTFLOAT> &TempLowerBounds);
	int ComputeAssignmentWithoutFilterICML15(const int i,const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row, long *DistanceCount);


public:
	YinyangKMABase(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,int at,bool IC);
	EXPFLOAT ComputeSSE(const Array<OPTFLOAT> &vec);
	virtual ~YinyangKMABase();

	virtual double ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount);
	virtual bool CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount);

	int GetT() const {return t;}
	CentroidVectorPermutation & GetPermutation() {return Perm;}

};



class YinyangKMA : public YinyangKMABase {

protected:


	virtual void DbgPrintInfo(int i,const Array<OPTFLOAT> &vec,const float * __restrict__ row);


public:
	YinyangKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,int at,bool IC);
	virtual void PrintNumaLocalityInfo();
	virtual ~YinyangKMA() {}
};



#endif /* CLUST_YINYANGKMA_H_ */
