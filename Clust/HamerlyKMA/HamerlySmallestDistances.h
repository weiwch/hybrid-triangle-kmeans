/*
 * HamerlySmallestDistances.h
 *
 *  Created on: Mar 2, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_HAMERLYKMA_HAMERLYSMALLESTDISTANCES_H_
#define CLUST_HAMERLYKMA_HAMERLYSMALLESTDISTANCES_H_
#include "../CentroidVector.h"

class HamerlySmallestDistances {

protected:
	int ncols;
	int nclusters;
	CentroidVector &CV;

public:
	HamerlySmallestDistances(CentroidVector &aCV);
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, ThreadPrivateVector<OPTFLOAT> &Distances)=0;

	virtual ~HamerlySmallestDistances() {}
};

class HamerlyOpenMPSmallestDistances : public HamerlySmallestDistances  {

public:
	HamerlyOpenMPSmallestDistances(CentroidVector &aCV);
	virtual void FillSmallestDistances(const Array<OPTFLOAT> &vec, ThreadPrivateVector<OPTFLOAT> &Distances);
};
#endif /* CLUST_HAMERLYKMA_HAMERLYSMALLESTDISTANCES_H_ */
