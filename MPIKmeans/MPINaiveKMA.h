/*
 * MPINaiveKMA.h
 *
 *  Created on: Oct 10, 2015
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPINAIVEKMA_H_
#define MPIKMEANS_MPINAIVEKMA_H_

#include "../Clust/KMAlgorithm.h"
#include "MPIForgyInitializer.h"
#include "MPIKMAReducer.h"
#include "DistributedNumaDataset.h"


class MPINaiveKMA: public NaiveKMA {

protected:

	MPIKMAReducer *pReducer;

public:
	MPINaiveKMA(CentroidVector &aCV,DistributedNumaDataset &D,CentroidRepair *pRepair,MPIKMAReducer *pR);

	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit);
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont);

	virtual ~MPINaiveKMA();
};

#endif /* MPIKMEANS_MPINAIVEKMA_H_ */
