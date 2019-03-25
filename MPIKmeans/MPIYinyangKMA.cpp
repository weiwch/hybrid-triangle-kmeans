/*
 * MPIYinyangKMA.cpp
 *
 *  Created on: Jan 24, 2016
 *      Author: wkwedlo
 */

#include "MPIYinyangKMA.h"

MPIYinyangKMA::MPIYinyangKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair,int at,bool IC,MPIKMAReducer *pR)
: YinyangKMA(aCV,D,pRepair,at,IC)
{
	pReducer=pR;
}

MPIYinyangKMA::~MPIYinyangKMA() {
}

void MPIYinyangKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit) {
	pReducer->ReduceData(Centers, Counts, Fit);
}

void MPIYinyangKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont) {
	pReducer->ReduceData(Centers, Counts, bCont);
}


