/*
 * MPIHamerlySmallestDistances.h
 *
 *  Created on: Mar 7, 2016
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPIHAMERLYKMA_MPIHAMERLYSMALLESTDISTANCES_H_
#define MPIKMEANS_MPIHAMERLYKMA_MPIHAMERLYSMALLESTDISTANCES_H_

#include "../../Clust/HamerlyKMA/HamerlySmallestDistances.h"
#include "../MPIItemDistribution.h"

class MPIHamerlySmallestDistances : public HamerlyOpenMPSmallestDistances {

protected:
	MPIItemDistribution Distribution;

public:
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, DynamicArray<OPTFLOAT> &Distances)=0;
	MPIHamerlySmallestDistances(CentroidVector &aCV);
	virtual ~MPIHamerlySmallestDistances();
};

class MPIHamerlyHierarchSD : public MPIHamerlySmallestDistances {
public:
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, DynamicArray<OPTFLOAT> &Distances);
	MPIHamerlyHierarchSD(CentroidVector &aCV);
};

class MPIHamerlyCrisscrossSD : public MPIHamerlySmallestDistances {
public:
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, DynamicArray<OPTFLOAT> &Distances);
	MPIHamerlyCrisscrossSD(CentroidVector &aCV);
};



#endif /* MPIKMEANS_MPIHAMERLYKMA_MPIHAMERLYSMALLESTDISTANCES_H_ */
