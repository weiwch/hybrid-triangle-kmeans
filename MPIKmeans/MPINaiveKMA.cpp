/*
 * MPINaiveKMA.cpp
 *
 *  Created on: Oct 10, 2015
 *      Author: wkwedlo
 */

#include <mpi.h>
#include <limits>
#include "MPINaiveKMA.h"
#include "MPIUtils.h"

/*
void MPINaiveKMA::PrintIterInfo(int i, double BestFit, double Rel,double Avoided) {
	int Rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	if (Rank==0)
		NaiveKMA::PrintIterInfo(i,BestFit,Rel,Avoided);
}
void MPINaiveKMA::PrintAvgAvoidance(double Avoided) {
	int Rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	if (Rank==0)
		NaiveKMA::PrintAvgAvoidance(Avoided);
}*/


void MPINaiveKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit, EXPFLOAT &Fit_pure) {
	pReducer->ReduceData(Centers,Counts,Fit,Fit_pure);
}

void MPINaiveKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont) {
	pReducer->ReduceData(Centers,Counts,bCont);
}


MPINaiveKMA::MPINaiveKMA(CentroidVector &aCV,DistributedNumaDataset &D,CentroidRepair *pRepair,MPIKMAReducer *pR) : NaiveKMA(aCV,D,pRepair) {
	pReducer=pR;
}

MPINaiveKMA::~MPINaiveKMA() {
}


