/*
 * MPIHamerlySmallestDistances.cpp
 *
 *  Created on: Mar 7, 2016
 *      Author: wkwedlo
 */

#include <limits>
#include <cmath>

#include "MPIHamerlySmallestDistances.h"
#include "../MPIUtils.h"

MPIHamerlySmallestDistances::MPIHamerlySmallestDistances(CentroidVector &aCV) : HamerlyOpenMPSmallestDistances(aCV),
							Distribution(CV.GetNClusters()) {
	TRACE4("Process %d out of %d first cluster %d cluster count %d\n",Distribution.GetRank(),
			Distribution.GetSize(),Distribution.GetFistItem(),Distribution.GetNItems());

}

MPIHamerlySmallestDistances::~MPIHamerlySmallestDistances() {
}

MPIHamerlyHierarchSD::MPIHamerlyHierarchSD(CentroidVector &aCV) : MPIHamerlySmallestDistances(aCV) {

}

void MPIHamerlyHierarchSD::FillSmallestDistances(const Array<OPTFLOAT> &vec, DynamicArray<OPTFLOAT> &Distances) {


	int Start=Distribution.GetFistItem();
	int Count=Distribution.GetNItems();
	OPTFLOAT distance;

#pragma omp parallel for default(none) shared(vec) private(distance) shared(Distances) firstprivate(Start,Count)
	for (int j = Start; j < Count; j++) {
		Distances[j]=std::numeric_limits<OPTFLOAT>::max();
		for (int i = 0; i < nclusters; i++) {
			if (j != i) {
				distance = CV.CentroidSquaredDistance(vec, i, j);

				if (distance < Distances[j]) {
					Distances[j] = distance;
				}
			}
		}
	}

	for (int i = Start; i < Count; i++) {
		Distances[i] = std::sqrt(Distances[i]);
	}
	MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,Distances.GetData(),Distribution.GetSubSizes(),Distribution.GetDisplacements(),
			OptFloatType(),MPI_COMM_WORLD);
}

MPIHamerlyCrisscrossSD::MPIHamerlyCrisscrossSD(CentroidVector &aCV) : MPIHamerlySmallestDistances(aCV) {

}


void MPIHamerlyCrisscrossSD::FillSmallestDistances(const Array<OPTFLOAT> &vec, DynamicArray<OPTFLOAT> &Distances) {


	int Start=Distribution.GetFistItem();
	int Count=Distribution.GetNItems();

	for (int j = Start; j < Count; j++) {
		OPTFLOAT MinDist=std::numeric_limits<OPTFLOAT>::max();

#pragma omp parallel for default(none) shared(vec,Distances) firstprivate(Start,Count,j)  reduction(min:MinDist)
		for (int i = 0; i < nclusters; i++) {
			if (j != i) {
				OPTFLOAT distance = CV.CentroidSquaredDistance(vec, i, j);

				if (distance < MinDist) {
					MinDist = distance;
				}
			}
		}
		Distances[j]=sqrt(MinDist);
	}

	MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,Distances.GetData(),Distribution.GetSubSizes(),Distribution.GetDisplacements(),
			OptFloatType(),MPI_COMM_WORLD);
}
