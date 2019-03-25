/*
 * MPICentroidRandomRepair.h
 *
 *  Created on: Feb 3, 2016
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPICENTROIDRANDOMREPAIR_H_
#define MPIKMEANS_MPICENTROIDRANDOMREPAIR_H_

#include "../Clust/CentroidRepair.h"
#include "DistributedNumaDataset.h"

class MPICentroidRandomRepair: public CentroidRepair {
	DistributedNumaDataset &Data;

	int FindRoot(int Size,int Source);
public:
	MPICentroidRandomRepair(DistributedNumaDataset &D,int ncl);
	virtual void RepairVec(Array<OPTFLOAT> &vec,int Pos);
	virtual ~MPICentroidRandomRepair() {}
};

#endif /* MPIKMEANS_MPICENTROIDRANDOMREPAIR_H_ */
