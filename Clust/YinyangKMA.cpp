/*
 * YinyangKMA.cpp
 *
 *  Created on: Nov 12, 2015
 *      Author: wkwedlo
 */

#include "YinyangKMA.h"
#include "PlusPlusInitializer/PlusPlusInitializer.h"
#include "../Util/StdOut.h"
#include "../Util/OpenMP.h"

#include <limits>
#include <cmath>
#include <sys/times.h>
#include <time.h>


//#define DUMP_GROUP_MAX
//#define TRACE_BOUNDS
//#define TRACE_FILTER
//#define TRACE_COUNTER

/* an i-th Group contains consecutive centroids. Centroid j is the member of the grooup i
 * if and only if GroupFirst[i] <= j < GroupNotLast[j]. Groups are mutually exclusive :)
 */
void YinyangKMABase::ComputeGroupSizes() {
	for (int Group=0;Group<t; Group++) {
		int SubSize=nclusters/t;
		int SubRemainder=nclusters % t;
		int Offset;
		if (Group<SubRemainder) {
			SubSize++;
			Offset=SubSize*Group;
		} else {
			Offset=SubSize*Group+SubRemainder;
		}
		GroupNotLast[Group]=Offset+SubSize;
		GroupFirst[Group]=Offset;
	}
	ComputeGroupNumbers();
}

void YinyangKMABase::ComputeGroupNumbers() {

	for(int Group=0;Group<t;Group++) {
		for(int j=GroupFirst[Group];j<GroupNotLast[Group];j++)
			GroupNumbers[j]=Group;
		TRACE3("YinyangKMA group %d first :%d notlast  %d\n",Group,GroupFirst[Group],GroupNotLast[Group]);
	}
}

YinyangKMABase::YinyangKMABase(CentroidVector &aCV, StdDataset &D, CentroidRepair *pR,int at,bool IC) :
		KMeansAlgorithm(aCV,D,pR),KMeansWithoutMSE(aCV, D, IterCount),Perm(aCV.GetNClusters())  {
	InitialCluster=IC;
	ICntr=0;
	int nRows=Data.GetRowCount();
	if (at>0)
		t=at;
	else
		t=nclusters/10;
	if (t<1) t=1;
	if (t>nclusters)
		t=nclusters;

	kma_printf("Number of cluster groups: %d\n",t);
	kma_printf("Initial clustering of centroids: %d\n",(int)InitialCluster);

	kma_printf("sizeof(YinyangThreadData)=%ld\n",sizeof(YinyangThreadData));

	Assignment.SetSize(nRows);
	UpperBounds.SetSize(nRows);
	LowerBounds.SetSize(nRows,t);

	GroupFirst.SetSize(t);
	GroupNotLast.SetSize(t);
	GroupMaxMoved.SetSize(t);
	GroupNumbers.SetSize(nclusters);
	Center.SetSize(ncols*nclusters);
	tmp_vec.SetSize(ncols*nclusters);
	Counts.SetSize(nclusters);
	DistanceMoved.SetSize(nclusters);

	pOMPReducer=new Log2OpenMPReducer(aCV);
	pClust=NULL;
	int NThreads=omp_get_max_threads();
	OMPData.SetSize(NThreads);
#pragma omp parallel
	{
		int i=omp_get_thread_num();
		OMPData[i].GroupBestIndex.SetSize(t);
		OMPData[i].GroupMask.SetSize(t);
		OMPData[i].TempLowerBounds.SetSize(t);
		OMPData[i].row.SetSize(ncols);
	}

}

YinyangKMABase::~YinyangKMABase() {
	delete pOMPReducer;
	if (pClust!=NULL)
		delete pClust;
}


// Step 0 not described in the paper
void YinyangKMABase::InitCentroids(const Array<OPTFLOAT> &vec) {

	pOMPReducer->ClearArrays(Center,Counts);

}


EXPFLOAT YinyangKMABase::ComputeSSE(const Array<OPTFLOAT> &vec) {
	EXPFLOAT ret = 0;
	const int rowCount = Data.GetRowCount();

#pragma omp parallel for default(none) firstprivate(rowCount) shared(vec) reduction(+:ret)
	for (int i = 0; i < rowCount; i++) {
		ret += CV.SquaredDistance(Assignment[i], vec, Data.GetRowNew(i));
	}

	return ret;
}

void YinyangKMABase::InitDataStructures(Array<OPTFLOAT> &vec) {
	InitCentroids(vec);
	if (InitialCluster)
		ClusterInitialCenters(vec);
	else
		ComputeGroupSizes();
	FirstIteration(vec);
	ICntr=0;
	pOMPReducer->ReduceToZero();
	bool bCont=true;
	ReduceMPIData(pOMPReducer->GetThreadCenter(0),pOMPReducer->GetThreadCounts(0),bCont);
}


void YinyangKMABase::ComputeNewVec(Array<OPTFLOAT> &vec) {
	tmp_vec=vec;

	pOMPReducer->AddZeroToCenter(Center,Counts);

	for(int i=0;i<nclusters;i++) {
		if (Counts[i]>0) {
			OPTFLOAT f=(OPTFLOAT)1.0/(OPTFLOAT)Counts[i];
			for(int j=i*ncols;j<(i+1)*ncols;j++)
				vec[j]=Center[j]*f;
		} else
			pRepair->RepairVec(vec,i);
	}
	// Drift of each center
	for (int i=0; i<nclusters; i++) {
		DistanceMoved[i] = std::sqrt(CV.SquaredDistance(tmp_vec, vec, i));
	}
}

void YinyangKMABase::ComputeGroupDrifts() {
	// Max drift in each group of centers
	for(int Group=0;Group<t;Group++) {
		OPTFLOAT GroupMax=(OPTFLOAT)0.0;
		for(int i=GroupFirst[Group];i<GroupNotLast[Group];i++)
			if (DistanceMoved[i]>GroupMax)
				GroupMax=DistanceMoved[i];
		GroupMaxMoved[Group]=GroupMax;
	}
#ifdef DUMP_GROUP_MAX
	for(int Group=0;Group<t;Group++) {
		TRACE2("Group%d Max Drift: %g\n",Group,GroupMaxMoved[Group]);
	}
#endif
}

void YinyangKMABase::MoveObject(const int PrevA, int NewA,const ThreadPrivateVector<OPTFLOAT> &row,int tid) {


	ThreadPrivateVector<OPTFLOAT> &DeltaCenter=pOMPReducer->GetThreadCenter(tid);
	ThreadPrivateVector<int> &DeltaCounts=pOMPReducer->GetThreadCounts(tid);
	CV.MoveRow(PrevA,NewA,DeltaCenter,row);
	DeltaCounts[PrevA]--;
	DeltaCounts[NewA]++;
}

double YinyangKMABase::ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount)
{
	EXPFLOAT SSE=ComputeSSE(vec);
	ComputeNewVec(vec);
	ReclusterCenters(vec);
	ComputeGroupDrifts();
	OuterLoop(vec,distanceCount);
	pOMPReducer->ReduceToZero();
	ReduceMPIData(pOMPReducer->GetThreadCenter(0),pOMPReducer->GetThreadCounts(0),SSE);
	ICntr++;
	return SSE/(EXPFLOAT)Data.GetTotalRowCount();
}


bool YinyangKMABase::CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount) {
	ComputeNewVec(vec);
	ReclusterCenters(vec);
	ComputeGroupDrifts();
	bool Cont=OuterLoop(vec,distanceCount);
	pOMPReducer->ReduceToZero();
	ReduceMPIData(pOMPReducer->GetThreadCenter(0),pOMPReducer->GetThreadCounts(0),Cont);
	ICntr++;
	return Cont;
}


void YinyangKMABase::ClusterCenters(Array<OPTFLOAT> &vec,CentroidVectorPermutation &aPerm) {
	StdDataset *pTempData = new StdDataset(vec,ncols,nclusters);

	DynamicArray<OPTFLOAT> initvec(t*ncols);

	for(int i=0;i<t;i++) {
		const DataRow &row=pTempData->GetRow(i);
		for(int j=0;j<ncols;j++)
			initvec[i*ncols+j]=row[j];
	}

	DynamicArray<int> ClNums(nclusters);
	pClust->FindAssignment(*pTempData,initvec,ClNums,t);
	int Counter=0;
	for(int Group=0;Group<t;Group++) {
		GroupFirst[Group]=Counter;
		for(int i=0;i<nclusters;i++)
			if (ClNums[i]==Group) {
				//kma_printf(" %d",i);
				aPerm.SetPermutationTarget(i,Counter);
				Counter++;
			}
		GroupNotLast[Group]=Counter;
		//kma_printf("\n");
	}
	ComputeGroupNumbers();
	aPerm.PermuteCentroidVector(vec,ncols);
	delete pTempData;
}


void YinyangKMABase::ClusterCentersUsingGroupNumbers(Array<OPTFLOAT> &vec,CentroidVectorPermutation &aPerm) {
	StdDataset *pTempData = new StdDataset(vec,ncols,nclusters);


	pClust->Recluster(*pTempData,GroupNumbers,t);

	DynamicArray<int> ClNums(nclusters);
	ClNums=GroupNumbers;

	int Counter=0;
	for(int Group=0;Group<t;Group++) {
		GroupFirst[Group]=Counter;
		for(int i=0;i<nclusters;i++)
			if (ClNums[i]==Group) {
				//kma_printf(" %d",i);
				aPerm.SetPermutationTarget(i,Counter);
				Counter++;
			}
		GroupNotLast[Group]=Counter;
		//kma_printf("\n");
	}
	ComputeGroupNumbers();
	aPerm.PermuteCentroidVector(vec,ncols);
	delete pTempData;
}


void YinyangKMABase::ClusterInitialCenters(Array<OPTFLOAT> &vec) {


	timespec tpstart_monotonic,tpend_monotonic;
	clock_gettime(CLOCK_MONOTONIC, &tpstart_monotonic);

	ClusterCenters(vec,Perm);
	clock_gettime(CLOCK_MONOTONIC, &tpend_monotonic);
	double extime2_monotonic=(double)tpend_monotonic.tv_sec-(double)tpstart_monotonic.tv_sec+1e-9*((double)tpend_monotonic.tv_nsec-(double)tpstart_monotonic.tv_nsec);
	kma_printf("Initial Yinyang clustering took: %g seconds\n",extime2_monotonic);

}
int YinyangKMABase::ComputeAssignmentInFirstIter(const int i,const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row, long *DistanceCount) {

	long Counter=*DistanceCount;


	const OPTFLOAT OldUpperBound=UpperBounds[i];
	OPTFLOAT UB=OldUpperBound;
	int bestj=0;
	int BestGroup=GroupNumbers[bestj];

	for(int Group=0;Group<t;Group++) {
		const int GF=GroupFirst[Group];
		const int GnL=GroupNotLast[Group];
		OPTFLOAT LBG=std::numeric_limits<OPTFLOAT>::max();
		for(int j=GF;j<GnL;j++) {
				double dist=sqrt(CV.SquaredDistance(j,vec,row));
				Counter++;
				if (dist<LBG) {
					if (dist<UB) {
						if (BestGroup==Group)
							LBG=UB;
						else
							LowerBounds(i,BestGroup)=UB;

						UB=dist;
						bestj=j;
						BestGroup=Group;
					} else {
						LBG=dist;
					}
				}
			}
		if (LBG==std::numeric_limits<OPTFLOAT>::max())
			LowerBounds(i,Group)=(OPTFLOAT)0.0;
		else
			LowerBounds(i,Group)=LBG;
	}
	UpperBounds[i]=UB;

	*DistanceCount=Counter;
	return bestj;
}



void YinyangKMABase::FirstIteration(const Array<OPTFLOAT> &vec) {
	int nRows=Data.GetRowCount();
	long Counter=0;


	for(int j=0;j<nclusters;j++) {
		DistanceMoved[j]=(OPTFLOAT)0.0;
	}

#pragma omp parallel default(none) shared(vec) firstprivate(nRows,Counter)
	{
		int tid=omp_get_thread_num();
		pOMPReducer->ClearThreadData(tid);

		ThreadPrivateVector<OPTFLOAT> &DeltaCenter=pOMPReducer->GetThreadCenter(tid);
		ThreadPrivateVector<int> &DeltaCounts=pOMPReducer->GetThreadCounts(tid);
		ThreadPrivateVector<OPTFLOAT> &tpRow=OMPData[tid].row;

#pragma omp for
		for(int i=0;i<nRows;i++) {
			const float * __restrict__ row=Data.GetRowNew(i);
			UpperBounds[i]=std::numeric_limits<OPTFLOAT>::max();
			CV.ConvertToOptFloat(tpRow,row);
			int A=ComputeAssignmentInFirstIter(i,vec,tpRow,&Counter);
			Assignment[i]=A;
			for(int j=0;j<ncols;j++)
				DeltaCenter[A*ncols+j]+=row[j];
			DeltaCounts[A]++;
		}
	}
}

int YinyangKMABase::ComputeAssignmentWithFilterICML15(const int i,const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row, long *DistanceCount,
		const DynamicArray<bool> &GroupMask, const DynamicArray<OPTFLOAT> &TempLowerBounds) {

	int bestj=Assignment[i];
	const int A=bestj;
	long Counter=*DistanceCount;


	const OPTFLOAT OldUpperBound=UpperBounds[i];
	OPTFLOAT UB=OldUpperBound;
	int BestGroup=GroupNumbers[bestj];

	for(int Group=0;Group<t;Group++)
		if (GroupMask[Group]) {
			const int GF=GroupFirst[Group];
			const int GnL=GroupNotLast[Group];
			OPTFLOAT LBG=std::numeric_limits<OPTFLOAT>::max();
			const OPTFLOAT TLB=TempLowerBounds[Group];
			for(int j=GF;j<GnL;j++) {
				if (j!=A && LBG > TLB - DistanceMoved[j]) {
					double dist=sqrt(CV.SquaredDistance(j,vec,row));
					Counter++;
					if (dist<LBG) {
						if (dist<UB) {
							if (BestGroup==Group)
								LBG=UB;
							else
								LowerBounds(i,BestGroup)=UB;

							UB=dist;
							bestj=j;
							BestGroup=Group;
						} else {
							LBG=dist;
						}
					}
				}
			}
			if (LBG==std::numeric_limits<OPTFLOAT>::max())
				LowerBounds(i,Group)=(OPTFLOAT)0.0;
			else
				LowerBounds(i,Group)=LBG;
		}


	const int OldGroup=GroupNumbers[A];
	//  A possible upper bound update (not mentioned in ICML'15 paper)
	if (OldUpperBound<LowerBounds(i,OldGroup) && OldUpperBound>UB)
		LowerBounds(i,OldGroup)=OldUpperBound;

	UpperBounds[i]=UB;

	*DistanceCount=Counter;
	return bestj;
}

int YinyangKMABase::ComputeAssignmentWithoutFilterICML15(const int i,const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row, long *DistanceCount) {
	int tid=omp_get_thread_num();
	DynamicArray<bool> &GroupMask=OMPData[tid].GroupMask;
	int bestj=Assignment[i];
	const int A=bestj;
	long Counter=*DistanceCount;


	const OPTFLOAT OldUpperBound=UpperBounds[i];
	OPTFLOAT UB=OldUpperBound;
	int BestGroup=GroupNumbers[bestj];

	for(int Group=0;Group<t;Group++)
		if (GroupMask[Group]) {
			const int GF=GroupFirst[Group];
			const int GnL=GroupNotLast[Group];
			OPTFLOAT LBG=std::numeric_limits<OPTFLOAT>::max();
			for(int j=GF;j<GnL;j++) {
				if (j!=A) {
					double dist=sqrt(CV.SquaredDistance(j,vec,row));
					Counter++;
					if (dist<LBG) {
						if (dist<UB) {
							if (BestGroup==Group)
								LBG=UB;
							else
								LowerBounds(i,BestGroup)=UB;

							UB=dist;
							bestj=j;
							BestGroup=Group;
						} else {
							LBG=dist;
						}
					}
				}
			}
			if (LBG==std::numeric_limits<OPTFLOAT>::max())
				LowerBounds(i,Group)=(OPTFLOAT)0.0;
			else
				LowerBounds(i,Group)=LBG;
		}


	const int OldGroup=GroupNumbers[A];
	//  A possible upper bound update (not mentioned in ICML'15 paper)
	if (OldUpperBound<LowerBounds(i,OldGroup) && OldUpperBound>UB)
		LowerBounds(i,OldGroup)=OldUpperBound;

	UpperBounds[i]=UB;

	*DistanceCount=Counter;
	return bestj;
}


void YinyangKMABase::DbgVerifyAssignment(int i,const Array<OPTFLOAT> &vec,const float * __restrict__ row) {
#ifdef _DEBUG
	double minssq=std::numeric_limits<OPTFLOAT>::max();
	int bestj=-1;
	for(int j=0;j<nclusters;j++) {
		double ssq=CV.SquaredDistance(j,vec,row);
		if (ssq<minssq) {
			bestj=j;
			minssq=ssq;
		}
	}
	if (bestj!=Assignment[i]) {
		printf("bestj: %d Assignment[i]: %d UpperBounds[i]: %g\n",bestj,Assignment[i],UpperBounds[i]);
		DbgPrintInfo(i,vec,row);
		ASSERT(false);
	}
#endif
}

void YinyangKMABase::DbgVerifyLowerBounds(int i,int bestj,const Array<OPTFLOAT> &vec,const float * __restrict__ row) {
#ifdef _DEBUG
	for(int Group=0;Group<t;Group++) {
		for(int j=GroupFirst[Group];j<GroupNotLast[Group];j++) {
			if (j!=bestj) {
				double dist=std::sqrt(CV.SquaredDistance(j,vec,row));
				if (dist<LowerBounds(i,Group)) {
					printf("Group%d Centroid%d Distance: %g LowerBound: %g\n",Group,j,dist,LowerBounds(i,Group));
					printf("bestj: %d Assignment[i]: %d UpperBounds[i]: %g\n",bestj,Assignment[i],UpperBounds[i]);
					DbgPrintInfo(i,vec,row);
					ASSERT(false);
				}
			}
		}
		if (LowerBounds(i,Group)==UpperBounds[i])
			ASSERT(false);
	}
#endif
}


void YinyangKMABase::DbgPrintAll(const Array<OPTFLOAT> &vec,int FilterRow) {
	int nRows=Data.GetRowCount();
	printf("Cluster drifts:");
	for(int j=0;j<nclusters;j++) {
		printf("% g",DistanceMoved[j]);
	}
	printf(" Group drifts:");
	for(int Group=0;Group<t;Group++)
		printf("% g",GroupMaxMoved[Group]);
	printf("\n");


	for(int i=0;i<nRows;i++) {
		if (FilterRow==-1 || FilterRow==i) {
			printf("i=%d LB:",i);
			for(int Group=0;Group<t;Group++)
				printf(" %g",LowerBounds(i,Group));
			printf(" UB: %g",UpperBounds[i]);
			const float * __restrict__ row = Data.GetRowNew(i);
			printf(" Distances:");
			for(int j=0;j<nclusters;j++) {
				double dist=std::sqrt(CV.SquaredDistance(j,vec,row));
				printf(" %g",dist);
			}
			printf("\n");
		}
	}
}




YinyangKMA::YinyangKMA(CentroidVector &aCV, StdDataset &D, CentroidRepair *pR,int at,bool IC) : YinyangKMABase(aCV,D,pR,at,IC) {
	pClust=new KMeansYinyangClusterer;
}















bool YinyangKMABase::OuterLoop(const Array<OPTFLOAT> &vec, long *DistanceCount) {

	int nRows=Data.GetRowCount();
	long Counter=0;
	bool Cont=false;

#pragma omp parallel default(none)  firstprivate(nRows) shared(vec) reduction(+:Counter) reduction(||:Cont)
	{
		int tid=omp_get_thread_num();
		pOMPReducer->ClearThreadData(tid);

		DynamicArray<bool> &GroupMask=OMPData[tid].GroupMask;
		DynamicArray<OPTFLOAT> &TempLowerBounds=OMPData[tid].TempLowerBounds;
		ThreadPrivateVector<OPTFLOAT> &tpRow=OMPData[tid].row;
#pragma omp for OMPDYNAMIC
		for(int i=0;i<nRows;i++) {
			const float * __restrict__ row=Data.GetRowNew(i);
			OPTFLOAT * __restrict__ LBRow=LowerBounds(i);

			/// Step 3.2 and 3.2a For each point update upperbound and group lower bounds. Find global lower bound
			UpperBounds[i]+=DistanceMoved[Assignment[i]];
			OPTFLOAT GlobalLowerBound=std::numeric_limits<OPTFLOAT>::max();
			for(int Group=0;Group<t;Group++) {
				TempLowerBounds[Group]=LBRow[Group];
				LBRow[Group]-=GroupMaxMoved[Group];
				if (LBRow[Group]<GlobalLowerBound)
					GlobalLowerBound=LBRow[Group];
			}
			CV.ConvertToOptFloat(tpRow,row);
	// Step 3.2b Check global bounds
			if (UpperBounds[i]<=GlobalLowerBound) {
				continue;
			}

	// Step 3.2c tighten upper bound by real distance
			UpperBounds[i]=sqrt(CV.SquaredDistance(Assignment[i],vec,tpRow));

			Counter++;
	// Step 3.2d check global bounds once again
			if (UpperBounds[i]<=GlobalLowerBound) {
				continue;
			}

	//  Step 3.2e Group filtering
			for(int Group=0;Group<t; Group++) {
				if (UpperBounds[i]<=LBRow[Group]) {
	// There is no need to consider this group at later stages
					GroupMask[Group]=false;
				} else {
	// Must consider this group at later stages
					GroupMask[Group]=true;
				}
			}
			int OldAssignment=Assignment[i];
			int NewAssignment=ComputeAssignmentWithFilterICML15(i,vec,tpRow,&Counter,GroupMask,TempLowerBounds);

			if (NewAssignment!=OldAssignment) {
				MoveObject(OldAssignment,NewAssignment,tpRow,tid);
				Assignment[i]=NewAssignment;
				Cont=true;
			}
		}
	}
	(*DistanceCount)+=Counter;
	return Cont;
}



void YinyangKMA::PrintNumaLocalityInfo() {
	double assignmentLoc = NumaLocalityofArray(Assignment) * 100.0;
	double upperBoundsLoc = NumaLocalityofArray(UpperBounds) * 100.0;
	double lowerBoundsLoc = NumaLocalityofMatrix(LowerBounds) * 100.0;
	kma_printf("NUMA locality of Assignment is %1.2f%%\n", assignmentLoc);
	kma_printf("NUMA locality of UpperBounds is %1.2f%%\n", upperBoundsLoc);
	kma_printf("NUMA locality of LowerBounds is %1.2f%%\n", lowerBoundsLoc);
#pragma omp parallel default(none)
	{
		int id=omp_get_thread_num();
		double centerLoc=NumaLocalityofPrivateArray(pOMPReducer->GetThreadCenter(id))*100.0;
		double countLoc=NumaLocalityofPrivateArray(pOMPReducer->GetThreadCounts(id))*100.0;
		double maskLoc=NumaLocalityofPrivateArray(OMPData[id].GroupMask)*100.0;
		double bestidxLoc=NumaLocalityofPrivateArray(OMPData[id].GroupBestIndex)*100.0;
		double tempLoc=NumaLocalityofPrivateArray(OMPData[id].TempLowerBounds)*100.0;


		kma_printf("Thread %d NUMA locality of private array OMPData[%d].DeltaCenter is %1.2f%%\n",id,id,centerLoc);
		kma_printf("Thread %d NUMA locality of private array OMPData[%d].DeltaCounts is %1.2f%%\n",id,id,countLoc);
		kma_printf("Thread %d NUMA locality of private array OMPData[%d].GroupMask is %1.2f%%\n",id,id,maskLoc);
		kma_printf("Thread %d NUMA locality of private array OMPData[%d].GroupBestIndex is %1.2f%%\n",id,id,bestidxLoc);
		kma_printf("Thread %d NUMA locality of private array OMPData[%d].TempLowerBounds is %1.2f%%\n",id,id,tempLoc);

	}

}


void YinyangKMA::DbgPrintInfo(int i,const Array<OPTFLOAT> &vec,const float * __restrict__ row) {
	printf("For i=%d GroupMasks/LowerBounds/GroupSecond:",i);

	int tid=omp_get_thread_num();
	DynamicArray<bool> &GroupMask=OMPData[tid].GroupMask;

	for(int Group=0;Group<t;Group++)
		printf(" (%d %g)",GroupMask[Group],LowerBounds(i,Group));
	printf("\nCentroid Distances:");
	for(int j=0;j<nclusters;j++) {
		double dist=std::sqrt(CV.SquaredDistance(j,vec,row));
		printf(" %d: %g",j,dist);
	}
	printf("\n");

}
