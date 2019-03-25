/*
 * CentroidVectorPermutation.cpp
 *
 *  Created on: Jan 31, 2016
 *      Author: wkwedlo
 */
#include <stdio.h>
#include "../Util/FileException.h"
#include "CentroidVectorPermutation.h"

CentroidVectorPermutation::CentroidVectorPermutation(int ncl) {
	nclusters=ncl;
	Perm.SetSize(nclusters);
	invPerm.SetSize(nclusters);
	for(int i=0;i<nclusters;i++)
		invPerm[i]=Perm[i]=-1;
}

void CentroidVectorPermutation::PermuteCentoridDrifts(DynamicArray<OPTFLOAT> &Drifts) {
	ASSERT(Drifts.GetSize()==nclusters);
	DynamicArray<OPTFLOAT> NewDrifts(nclusters);
	for(int i=0;i<nclusters;i++) {
		NewDrifts[Perm[i]]=Drifts[i];
	}
	Drifts=NewDrifts;
}


void CentroidVectorPermutation::PermuteCentroidVector(Array<OPTFLOAT> &vec,int ncols) {
	ASSERT(vec.GetSize()==ncols*nclusters);
	DynamicArray<OPTFLOAT> newvec(ncols*nclusters);
	for(int i=0;i<nclusters;i++) {
		int Target=Perm[i];
		for(int j=0;j<ncols;j++)
			newvec[Target*ncols+j]=vec[i*ncols+j];
	}
	vec=newvec;
}

void CentroidVectorPermutation::PermuteCentroidVector(ThreadPrivateVector<OPTFLOAT> &vec,int ncols) {
	ASSERT(vec.GetSize()==ncols*nclusters);
	DynamicArray<OPTFLOAT> newvec(ncols*nclusters);
	for(int i=0;i<nclusters;i++) {
		int Target=Perm[i];
		for(int j=0;j<ncols;j++)
			newvec[Target*ncols+j]=vec[i*ncols+j];
	}
	for(int i=0;i<nclusters*ncols;i++)
		vec[i]=newvec[i];
}

void CentroidVectorPermutation::PermuteCentroidCounts(ThreadPrivateVector<int> &Counts) {
	ASSERT(Counts.GetSize()==nclusters);
	DynamicArray<int> NewCounts(nclusters);
	for(int i=0;i<nclusters;i++) {
		NewCounts[Perm[i]]=Counts[i];
	}
	for(int i=0;i<nclusters;i++)
		Counts[i]=NewCounts[i];

}

void CentroidVectorPermutation::ComputeInverse() {
	for(int i=0;i<nclusters;i++)
		invPerm[Perm[i]]=i;
}

bool CentroidVectorPermutation::IsIdentity() {
	for(int i=0;i<nclusters;i++) {
		if (Perm[i]!=i)
			return false;
	}
	return true;
}

void CentroidVectorPermutation::Dump() {
	printf("Permutation:\n");
	for(int i=0;i<nclusters;i++) {
		printf("%d --> %d\n",i,Perm[i]);
	}
	printf("Inverse permutation:\n");
	for(int i=0;i<nclusters;i++) {
		printf("%d --> %d\n",i,invPerm[i]);
	}
}

void CentroidVectorPermutation::Write(char *fname) {
	FILE *pF=fopen(fname,"wb");
	if (pF==NULL)
		throw FileException("Cannot open permutation file for writing");
	fwrite(Perm.GetData(),sizeof(int),nclusters,pF);
	fclose(pF);
}

void CentroidVectorPermutation::Read(char *fname) {
	FILE *pF=fopen(fname,"rb");
	if (pF==NULL)
		throw FileException("Cannot open permutation file for reading");
	fread(Perm.GetData(),sizeof(int),nclusters,pF);
	fclose(pF);
}


CentroidVectorPermutation::~CentroidVectorPermutation() {
}

