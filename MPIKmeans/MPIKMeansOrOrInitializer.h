/*
 * MPIKMeansOrOrInitializer.h
 *
 *  Created on: May 15, 2017
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPIKMEANSORORINITIALIZER_H_
#define MPIKMEANS_MPIKMEANSORORINITIALIZER_H_

#include "../Clust/KMeansOrOrInitializer.h"
#include "DistributedNumaDataset.h"

class MPIKMeansOrOrInitializer: public KMeansOrOrInitializer {
public:
	MPIKMeansOrOrInitializer(DistributedNumaDataset &D,CentroidVector &aCV,int cl);
	virtual ~MPIKMeansOrOrInitializer();
};

#endif /* MPIKMEANS_MPIKMEANSORORINITIALIZER_H_ */
