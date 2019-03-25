/*
 * ElkanSmallestDistances.h
 *
 *  Created on: Mar 2, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_ELKANKMA_ELKANSMALLESTDISTANCES_H_
#define CLUST_ELKANKMA_ELKANSMALLESTDISTANCES_H_

#include "../CentroidVector.h"
#include "../../Util/LargeVector.h"


class ElkanSmallestDistances {

protected:
	int ncols;
	int nclusters;
	CentroidVector &CV;


public:
	ElkanSmallestDistances(CentroidVector &aCV);
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,DynamicArray<OPTFLOAT> &SmallestDistances)=0;
	virtual ~ElkanSmallestDistances();
};


class ElkanOpenMPSmallestDistances : public ElkanSmallestDistances {

public:
	ElkanOpenMPSmallestDistances(CentroidVector &aCV);
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,DynamicArray<OPTFLOAT> &SmallestDistances);
};


#endif /* CLUST_ELKANKMA_ELKANSMALLESTDISTANCES_H_ */
