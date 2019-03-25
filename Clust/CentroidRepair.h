/*
 * CentroidRepair.h
 *
 *  Created on: Feb 3, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_CENTROIDREPAIR_H_
#define CLUST_CENTROIDREPAIR_H_

#include "../Util/StdDataset.h"

class CentroidRepair {
protected:
	int nclusters;
public:
	virtual void RepairVec(Array<OPTFLOAT> &vec,int Pos)=0;
	CentroidRepair(int ncl);
	virtual ~CentroidRepair() {}
};


class CentroidRandomRepair : public CentroidRepair {
protected:
	StdDataset &Data;
public:
	CentroidRandomRepair(StdDataset &Data,int ncl);
	virtual void RepairVec(Array<OPTFLOAT> &vec,int Pos);
};

class CentroidDeterministicRepair : public CentroidRepair {
protected:
	StdDataset &Data;
	int Next;
public:
	CentroidDeterministicRepair(StdDataset &Data,int ncl);
	virtual void RepairVec(Array<OPTFLOAT> &vec,int Pos);
};

#endif /* CLUST_CENTROIDREPAIR_H_ */
