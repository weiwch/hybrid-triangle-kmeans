/*
 * KmeansOrOrInitializer.h
 *
 *  Created on: Nov 30, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_KMEANSORORINITIALIZER_H_
#define CLUST_KMEANSORORINITIALIZER_H_

#include "KMeansInitializer.h"
#include "../Util/NumaDataset.h"


struct OrOrThreadData {
	DynamicArray< DynamicArray<float> > NewCenters;
};


class KMeansOrOrInitializer: public KMeansInitializer {

protected:

	double Oversample;
	int Rounds;

	NumaDataset &NumaData;

	LargeVector<OPTFLOAT> ClosestDistances;
	LargeVector<int> ClosestCentroids;

	DynamicArray< DynamicArray<float> > Centers;
	DynamicArray< DynamicArray<float> > NewCenters;


	DynamicArray< OPTFLOAT > Weights;

	DynamicArray <OrOrThreadData> ThreadData;

	void dbgDumpCentersAndWeights();

	void ComputeWeights();
	void ScanDataset(OPTFLOAT Cost);
	void InitDataStructures();
	void InsertCentroid(DynamicArray<float> &row);
	void InsertCentroids();
	void MergeThreadCenters();
	OPTFLOAT ComputeCost();

public:
	KMeansOrOrInitializer(NumaDataset &D,CentroidVector &aCV,int cl,double Os);
	virtual ~KMeansOrOrInitializer() {}
	virtual void Init(Array<OPTFLOAT> &v);

};

#endif /* CLUST_KMEANSORORINITIALIZER_H_ */
