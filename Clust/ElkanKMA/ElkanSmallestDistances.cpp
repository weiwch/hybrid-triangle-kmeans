/*
 * ElkanSmallestDistances.cpp
 *
 *  Created on: Mar 2, 2016
 *      Author: wkwedlo
 */
#include <limits>
#include <cmath>

#include "ElkanSmallestDistances.h"

ElkanSmallestDistances::ElkanSmallestDistances(CentroidVector &aCV) : CV(aCV){
	ncols=aCV.GetNCols();
	nclusters=aCV.GetNClusters();
}

ElkanSmallestDistances::~ElkanSmallestDistances() {
}

ElkanOpenMPSmallestDistances::ElkanOpenMPSmallestDistances(CentroidVector &aCV) : ElkanSmallestDistances(aCV) {

}

//#define __DUMP_DISTANCES


void ElkanOpenMPSmallestDistances::FillSmallestDistances(const Array<OPTFLOAT> &vec, Matrix<OPTFLOAT> &Distances,
				DynamicArray<OPTFLOAT> &SmallestDistances) {

#pragma omp parallel for default(none) shared(vec) schedule(static, 1) shared(Distances)
	for (int i = 0; i < nclusters; i++) {
		for (int j = i + 1; j < nclusters; j++) {
			Distances(i, j) = Distances(j, i) = std::sqrt(CV.CentroidSquaredDistance(vec, i, j)) / 2.0;
		}
	}

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
#ifdef __DUMP_DISTANCES
	printf("Distance matrix\n");
	for(int i=0;i<nclusters;i++) {
		for(int j=0;j<nclusters;j++)
			printf("%5.7f ",Distances(i,j));
		printf("\n");
	}
	printf("Smallest distance vector\n");
	for(int i=0;i<nclusters;i++)
		printf("%5.7f ",SmallestDistances[i]);
	printf("\n");
#endif
}


