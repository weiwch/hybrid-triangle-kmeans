/*
 * MPIYinyangKMA.h
 *
 *  Created on: Jan 24, 2016
 *      Author: wkwedlo
 */

#include "../Clust/YinyangKMA.h"
#include "MPIKMAReducer.h"
#include "DistributedNumaDataset.h"

#ifndef MPIKMEANS_MPIYINYANGKMA_H_
#define MPIKMEANS_MPIYINYANGKMA_H_

class MPIYinyangKMA: public YinyangKMA {

protected:
	MPIKMAReducer *pReducer;

	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit);
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont);

public:
	MPIYinyangKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair,int at,bool IC,MPIKMAReducer *pR);
	virtual ~MPIYinyangKMA();
};

#endif /* MPIKMEANS_MPIYINYANGKMA_H_ */
