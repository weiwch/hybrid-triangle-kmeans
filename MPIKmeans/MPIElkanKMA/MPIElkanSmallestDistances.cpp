/*
 * MPIElkanSmallestDistances.cpp
 *
 *  Created on: Apr 12, 2016
 *      Author: wkwedlo
 */


#include <mpi.h>
#include <limits>
#include <cmath>
#include "../MPIUtils.h"

#include "MPIElkanSmallestDistances.h"

MPIElkanSmallestDistances::MPIElkanSmallestDistances(CentroidVector &aCV) :
	ElkanSmallestDistances(aCV),Distribution(CV.GetNClusters()) {

	TRACE4("Process %d out of %d first cluster %d cluster count %d\n",Distribution.GetRank(),
			Distribution.GetSize(),Distribution.GetFistItem(),Distribution.GetNItems());

	int Size=Distribution.GetSize();
	SubSizes.SetSize(Size);
	Displacements.SetSize(Size);
	int Stride=aCV.GetNClusters();

	for(int i=0;i<Size;i++) {
		SubSizes[i]=Distribution.GetSubSizes()[i]*Stride;
		Displacements[i]=Distribution.GetDisplacements()[i]*Stride;
	}
}

MPIElkanSmallestDistances::~MPIElkanSmallestDistances() {

}

MPIElkanHierarchSD::MPIElkanHierarchSD(CentroidVector &aCV) : MPIElkanSmallestDistances(aCV) {

}

void MPIElkanHierarchSD::FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,
		DynamicArray<OPTFLOAT> &SmallestDistances) {

	int Start=Distribution.GetFistItem();
	int Count=Distribution.GetNItems();

#pragma omp parallel for default(none) shared(vec,Distances) firstprivate(Start,Count)
	for (int i = Start; i < Count; i++) {
		for (int j = 0; j < nclusters; j++) {
			if (i!=j)
				Distances(i, j) = std::sqrt(CV.CentroidSquaredDistance(vec, i, j)) / 2.0;
		}
	}

	MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,Distances.GetData(),Distribution.GetSubSizes(),Distribution.GetDisplacements(),
			OptFloatType(),MPI_COMM_WORLD);

#pragma omp parallel for default(none) shared(vec,Distances,SmallestDistances)
	for (int i = 0; i < nclusters; i++) {
		SmallestDistances[i]=std::numeric_limits<OPTFLOAT>::max();
		for (int j=0; j < nclusters; j++) {
			if (i!=j) {
				OPTFLOAT distance = Distances(i, j);
				if (distance < SmallestDistances[i]) {
					SmallestDistances[i] = distance;
				}
			}
		}
	}
}


MPIElkanCrisscrossSD::MPIElkanCrisscrossSD(CentroidVector &aCV) : MPIElkanSmallestDistances(aCV) {

}


void MPIElkanCrisscrossSD::FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,
		DynamicArray<OPTFLOAT> &SmallestDistances) {

	int Start=Distribution.GetFistItem();
	int Count=Distribution.GetNItems();

#pragma omp parallel default(none) shared(vec,Distances) firstprivate(Start,Count)
	for (int i = Start; i < Count; i++) {
#pragma omp for
		for (int j = 0; j < nclusters; j++) {
			if (i!=j)
				Distances(i, j) = std::sqrt(CV.CentroidSquaredDistance(vec, i, j)) / 2.0;
		}
	}

	MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,Distances.GetData(),Distribution.GetSubSizes(),Distribution.GetDisplacements(),
			OptFloatType(),MPI_COMM_WORLD);

#pragma omp parallel for default(none) shared(vec,Distances,SmallestDistances)
	for (int i = 0; i < nclusters; i++) {
		SmallestDistances[i]=std::numeric_limits<OPTFLOAT>::max();
		for (int j=0; j < nclusters; j++) {
			if (i!=j) {
				OPTFLOAT distance = Distances(i, j);
				if (distance < SmallestDistances[i]) {
					SmallestDistances[i] = distance;
				}
			}
		}
	}
}


