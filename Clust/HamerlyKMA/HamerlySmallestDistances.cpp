/*
 * HamerlySmallestDistances.cpp
 *
 *  Created on: Mar 2, 2016
 *      Author: wkwedlo
 */

#include <limits>
#include <cmath>

#include "HamerlySmallestDistances.h"

HamerlySmallestDistances::HamerlySmallestDistances(CentroidVector &aCV) : CV(aCV) {
	ncols=aCV.GetNCols();
	nclusters=aCV.GetNClusters();
}

HamerlyOpenMPSmallestDistances::HamerlyOpenMPSmallestDistances(CentroidVector &aCV) : HamerlySmallestDistances(aCV) {

}

void HamerlyOpenMPSmallestDistances::FillSmallestDistances(const Array<OPTFLOAT> &vec, ThreadPrivateVector<OPTFLOAT> &Distances) {
	OPTFLOAT distance;

#pragma omp parallel for default(none) shared(vec) private(distance) shared(Distances)
	for (int j = 0; j < nclusters; j++) {
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

	for (int i = 0; i < nclusters; i++) {
		Distances[i] = std::sqrt(Distances[i]);
	}

}

