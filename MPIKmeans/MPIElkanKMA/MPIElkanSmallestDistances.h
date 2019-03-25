/*
 * MPIElkanSmallestDistances.h
 *
 *  Created on: Apr 12, 2016
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPIELKANKMA_MPIELKANSMALLESTDISTANCES_H_
#define MPIKMEANS_MPIELKANKMA_MPIELKANSMALLESTDISTANCES_H_

#include "../../Clust/ElkanKMA/ElkanSmallestDistances.h"
#include "../MPIItemDistribution.h"


class MPIElkanSmallestDistances: public ElkanSmallestDistances {

protected:
	MPIItemDistribution Distribution;

	DynamicArray<int> SubSizes;
	DynamicArray<int> Displacements;

public:
	MPIElkanSmallestDistances(CentroidVector &aCV);
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,DynamicArray<OPTFLOAT> &SmallestDistances)=0;
	virtual ~MPIElkanSmallestDistances();
};

class  MPIElkanHierarchSD : public  MPIElkanSmallestDistances {
public:
	MPIElkanHierarchSD(CentroidVector &aCV);
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,DynamicArray<OPTFLOAT> &SmallestDistances);

};

class  MPIElkanCrisscrossSD : public  MPIElkanSmallestDistances {
public:
	MPIElkanCrisscrossSD(CentroidVector &aCV);
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,DynamicArray<OPTFLOAT> &SmallestDistances);

};


#endif /* MPIKMEANS_MPIELKANKMA_MPIELKANSMALLESTDISTANCES_H_ */
