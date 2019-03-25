/*
 * MPIItemDistribution.h
 *
 *  Created on: Mar 7, 2016
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPIITEMDISTRIBUTION_H_
#define MPIKMEANS_MPIITEMDISTRIBUTION_H_

#include "../Util/Array.h"

class MPIItemDistribution {
	int nItems;
	int Rank;
	int Size;
	int SubSize;
	int Offset;

	DynamicArray<int> SubSizes;
	DynamicArray<int> Displacements;

	void BuildSubSizes();

public:

	const int *GetSubSizes() {return SubSizes.GetData();}
	const int *GetDisplacements() {return Displacements.GetData();}
	int GetNItems() const {return SubSize;}
	int GetFistItem() const {return Offset;}
	int GetRank() const {return Rank;}
	int GetSize() const {return Size;}

	MPIItemDistribution(int ni);
	virtual ~MPIItemDistribution();
};

#endif /* MPIKMEANS_MPIITEMDISTRIBUTION_H_ */
