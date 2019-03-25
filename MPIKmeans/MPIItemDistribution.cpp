/*
 * MPIItemDistribution.cpp
 *
 *  Created on: Mar 7, 2016
 *      Author: wkwedlo
 */

#include <mpi.h>

#include "MPIItemDistribution.h"

MPIItemDistribution::MPIItemDistribution(int ni) {
	nItems=ni;
	MPI_Comm_size(MPI_COMM_WORLD,&Size);
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);

	SubSize=nItems/Size;
	int SubRemainder=nItems % Size;
	if (Rank<SubRemainder) {
		SubSize++;
		Offset=SubSize*Rank;
	} else {
		Offset=SubSize*Rank+SubRemainder;
	}
	SubSizes.SetSize(Size);
	Displacements.SetSize(Size);
	BuildSubSizes();
}

void MPIItemDistribution::BuildSubSizes() {
	int aSize=nItems/Size;
	int SubRemainder=nItems % Size;

	int Sum=0;
	for (int i=0;i<Size;i++) {
		if (i<SubRemainder)
			SubSizes[i]=aSize+1;
		else
			SubSizes[i]=aSize;
		Displacements[i]=Sum;
		Sum+=SubSizes[i];
	}
}


MPIItemDistribution::~MPIItemDistribution() {
	// TODO Auto-generated destructor stub
}

