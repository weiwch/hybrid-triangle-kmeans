/*
 * DistributedNumaDataset.h
 *
 *  Created on: Oct 8, 2015
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_DISTRIBUTEDNUMADATASET_H_
#define MPIKMEANS_DISTRIBUTEDNUMADATASET_H_

#include "../Util/NumaDataset.h"

class DistributedNumaDataset: public NumaDataset {
protected:

	DynamicArray<int> Offsets;
	DynamicArray<int> Counts;
	void ComputeDataPositions(int Size);
public:
	virtual void GlobalFetchRow(int Position,DataRow &row);
	virtual void Load(char *fname);
	DistributedNumaDataset();
	virtual ~DistributedNumaDataset() {}
};

#endif /* MPIKMEANS_DISTRIBUTEDNUMADATASET_H_ */
