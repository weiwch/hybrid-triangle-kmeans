/*
 * CentroidVectorPermutation.h
 *
 *  Created on: Jan 31, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_CENTROIDVECTORPERMUTATION_H_
#define CLUST_CENTROIDVECTORPERMUTATION_H_

#include "../Util/Array.h"
#include "../Util/LargeVector.h"

class CentroidVectorPermutation {
protected:
	int nclusters;
	DynamicArray<int> Perm;
	DynamicArray<int> invPerm;
public:
	void SetPermutationTarget(int i,int j) {Perm[i]=j;}
	int GetPermutationTarget(int i) const {return Perm[i];}
	int GetInversePermutationTarget(int i) const {return invPerm[i];}
	CentroidVectorPermutation(int ncl);
	void Write(char *fname);
	void Read(char *fname);
	void Dump();
	void ComputeInverse();
	void PermuteCentroidVector(ThreadPrivateVector<OPTFLOAT> &vec,int ncols);
	void PermuteCentroidVector(Array<OPTFLOAT> &vec,int ncols);
	void PermuteCentoridDrifts(DynamicArray<OPTFLOAT> &Drifts);
	void PermuteCentroidCounts(ThreadPrivateVector<int> &Counts);
	bool IsIdentity();
	virtual ~CentroidVectorPermutation();
};

#endif /* CLUST_CENTROIDVECTORPERMUTATION_H_ */
