/*
 * SameSizeKMA.cpp
 *
 *  Created on: Nov 8, 2016
 *      Author: wkwedlo
 */

#include <limits>
#include "SameSizeKMA.h"

//#define __TRACE_SAMESIZE

SameSizeKMA::SameSizeKMA(CentroidVector &aCV,StdDataset &D) : CV(aCV), Data(D) {

	nclusters=CV.GetNClusters();
	ncols=CV.GetNCols();
	pRepair=new CentroidDeterministicRepair(Data,nclusters);
	pKMA=new NaiveKMA(aCV,Data,pRepair);
}

SameSizeKMA::~SameSizeKMA() {
	delete pRepair;
	delete pKMA;
}
OPTFLOAT SameSizeKMA::FindAssignment(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment) {
	int nRows=Data.GetRowCount();
	int MaxObjCount =nRows % nclusters ? nRows/nclusters +1 : nRows/nclusters;

#ifdef __TRACE_SAMESIZE
	TRACE1("Max %d objects in a cluster\n",MaxObjCount);
#endif

	DynamicArray<OPTFLOAT> SquaredDists(nRows);
	DynamicArray<int> Assignment(nRows);
	DynamicArray<int> Counts(nclusters);

	OPTFLOAT sqSum=0.0;
	for(int i=0;i<nclusters;i++)
		Counts[i]=0;

	CV.ClassifyDataset(vec,Data,Assignment,SquaredDists);
	for(int i=0;i<nRows;i++) {
		PointDistance PD;
		PD.Point=i;
		PD.Distance=SquaredDists[i];
		PD.Centroid=Assignment[i];
		NewAssignment[i]=-1;
		heap.push(PD);
	}

	while(!heap.empty()) {
		PointDistance PD=heap.top();
		heap.pop();
		if (Counts[PD.Centroid]<MaxObjCount) {
#ifdef __TRACE_SAMESIZE
			TRACE3("Point %d with distance %5.3f was assigned to the cluster %d\n",PD.Point,PD.Distance,PD.Centroid);
#endif
			NewAssignment[PD.Point]=PD.Centroid;
			sqSum+=PD.Distance;
			Counts[PD.Centroid]++;
		} else {
#ifdef __TRACE_SAMESIZE
			TRACE3("Point %d with distance %5.3f was not assigned to the cluster %d\n",PD.Point,PD.Distance,PD.Centroid);
#endif
			OPTFLOAT minsq=std::numeric_limits<OPTFLOAT>::max();
			int optk=-1;
			for(int k=0;k<nclusters;k++) {
				double sqdist=CV.SquaredDistance(k,vec,Data.GetRow(PD.Point));
				if (Counts[k]<MaxObjCount) {
#ifdef __TRACE_SAMESIZE
					TRACE2("Can be assigned to the cluster %d with distance %5.3f\n",k,sqdist);
#endif
					if (sqdist<minsq) {
						optk=k;
						minsq=sqdist;
					}

				} else {
#ifdef __TRACE_SAMESIZE
					TRACE2("Cannot be assigned to the cluster %d with distance %5.3f\n",k,sqdist);
#endif
				}

			}
#ifdef __TRACE_SAMESIZE
			TRACE1("Should be assigned to the centroid %d -> reinserting to heap\n",optk);
#endif
			PD.Centroid=optk;
			PD.Distance=minsq;
			heap.push(PD);
		}
	}
#ifdef __TRACE_SAMESIZE
	TRACE1("After k-means clustering initial MSE is %5.4f\n",StartMSE);
	TRACE1("After points redistirbution MSE is %5.4f\n",sqSum/(OPTFLOAT)nRows);
	TRACE1("Verification of MSE  based on Assigment is %5.4f\n",CV.ComputeMSE(vec,Data,NewAssignment));
	for(int i=0;i<nRows;i++)
		Assignment[i]=Rand()*nclusters;
	TRACE1("Verification of MSE  based on random cluster assigment is %5.4f\n",CV.ComputeMSE(vec,Data,Assignment));
#endif
	return sqSum/(OPTFLOAT)nRows;
}

OPTFLOAT SameSizeKMA::Train(Array<OPTFLOAT> &vec,	DynamicArray<int> &NewAssignment) {
	double StartMSE=pKMA->RunKMeans(vec,4);
	FindAssignment(vec,NewAssignment);
	return StartMSE;

}
int SameSizeKMA::FindTransfers(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment) {
	int nRows=Data.GetRowCount();
	int Cntr=0;

	DynamicArray<bool> RowMovementMask(nRows);
	Matrix<OPTFLOAT> Distances(nRows,nclusters);

#pragma omp parallel for default(none) shared(vec,Distances,NewAssignment,RowMovementMask) firstprivate(nRows)
	for(int i=0;i<nRows;i++) {
		int bestk=-1;
		OPTFLOAT bestsq=std::numeric_limits<OPTFLOAT>::max(),assignsq;
		const DataRow &row=Data.GetRow(i);
		for (int k=0;k<nclusters;k++) {
			OPTFLOAT sqdist=CV.SquaredDistance(k,vec,row);
			Distances(i,k)=sqdist;
			if (k==NewAssignment[i]) {
				assignsq=sqdist;
			}
			if (sqdist<bestsq) {
				bestsq=sqdist;
				bestk=k;
			}
		}
		if (bestsq<assignsq)
			RowMovementMask[i]=true;
		else
			RowMovementMask[i]=false;
	}

	for(int i=0;i<nRows;i++) {
		if (!RowMovementMask[i])
			continue;
		const DataRow &rowi=Data.GetRow(i);
		for(int j=i+1;j<nRows;j++) {

			OPTFLOAT oldsqi=Distances(i,NewAssignment[i]);
			OPTFLOAT oldsqj=Distances(j,NewAssignment[j]);

			OPTFLOAT newsqi=Distances(i,NewAssignment[j]);
			OPTFLOAT newsqj=Distances(j,NewAssignment[i]);
			if (newsqi+newsqj<oldsqi+oldsqj) {
				int temp=NewAssignment[i];
				NewAssignment[i]=NewAssignment[j];
				NewAssignment[j]=temp;
				Cntr++;
				break;
			}
		}
	}
	return Cntr;
}

OPTFLOAT SameSizeKMA::TrainIterative(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment,int Iters,int Verbosity) {
	double MSE=pKMA->RunKMeans(vec,4);
	if (Verbosity>0)
		printf("Kmeans MSE: %5.3f\n",MSE);
	MSE=FindAssignment(vec,NewAssignment);
	if (Verbosity>0)
		printf("MSE after initit: %5.3f\n",MSE);
	for(int i=0;i<Iters;i++) {
		int Transfers=FindTransfers(vec,NewAssignment);
		if (Verbosity>0)
			printf("MSE in iter %d: %5.3f (%d transfers)\n",i+1,CV.ComputeMSE(vec,Data,NewAssignment),Transfers);
		CV.ComputeCenters(vec,Data,NewAssignment);
		if (Transfers==0)
			break;
	}
	if (Verbosity>1)
		DumpClusters(vec,NewAssignment);
	return MSE;
}

OPTFLOAT SameSizeKMA::TrainIterativeFromAssignment(Array<OPTFLOAT> &vec,DynamicArray<int> &NewAssignment,int Iters,int Verbosity) {
	if (Verbosity>0)
		printf("Starting MSE: %5.3f",CV.ComputeMSE(vec,Data,NewAssignment));
	for(int i=0;i<Iters;i++) {
		int Transfers=FindTransfers(vec,NewAssignment);
		if (Verbosity>0)
			printf("MSE in iter %d: %5.3f (%d transfers)\n",i+1,CV.ComputeMSE(vec,Data,NewAssignment),Transfers);
		CV.ComputeCenters(vec,Data,NewAssignment);
		if (Transfers==0)
			break;
	}
	return 0.0;

}



void SameSizeKMA::DumpClusters(Array<OPTFLOAT> &vec,DynamicArray<int> Assignment) {
	int nRows=Data.GetRowCount();
	DynamicArray<int> Counters(nclusters);
	for(int i=0;i<nclusters;i++)
		Counters[i]=0;

	for(int i=0;i<nRows;i++)
		Counters[Assignment[i]]++;

	for(int k=0;k<nclusters;k++) {
		for(int i=0;i<ncols;i++)
			printf("%1.3f ",vec[k*ncols+i]);
		printf(":%d Objects\n",Counters[k]);
	}
}
