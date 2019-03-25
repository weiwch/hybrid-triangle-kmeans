#include "PlusPlusInitializer.h"
#include "../../Util/Rand.h"
#include <limits>
#include <cmath>

PlusPlusInitializer::PlusPlusInitializer(StdDataset &D, CentroidVector &aCV, int ncl) :
		KMeansInitializer(D, aCV, ncl) {
	// TODO: GetTotalRowCount?
	int nRows=D.GetRowCount();
	ClosestDistances.SetSize(nRows);
	AggregatedSum.SetSize(nRows);
	Weights.SetSize(nRows);

	#pragma omp parallel for default(none) firstprivate(nRows)
	for (int i = 0; i < nRows; i++) {
		Weights[i]=(OPTFLOAT)1.0;
	}



}

void PlusPlusInitializer::SetWeights(DynamicArray<OPTFLOAT> aWeights) {
	int nRows=Data.GetRowCount();

#pragma omp parallel for default(none) firstprivate(nRows) shared(aWeights)
	for (int i = 0; i < nRows; i++) {
		Weights[i]=aWeights[i];
	}

}

void PlusPlusInitializer::Init(Array<OPTFLOAT> &vec) {

	// Choose one center uniformly at random from all data points
	int nRows=Data.GetRowCount();

	int randomIndex = RandInt(nRows-1);
	const float* __restrict firstCentroid = Data.GetRowNew(randomIndex);

	for (int i = 0; i < ncols; i++) {
		vec[i] = firstCentroid[i];
	}

#pragma omp parallel for default(none) firstprivate(nRows)
	for (int i = 0; i < nRows; i++) {
		ClosestDistances[i] = std::numeric_limits<OPTFLOAT>::infinity();
	}

	// Repeat until all centers have been chosen
	for (int i = 1; i < nclusters; i++) {

		// For each data point x, compute D(x) - the distance between x and the nearest center that has already been chosen
#pragma omp parallel for default(none) firstprivate(nRows,i) shared(vec)
		for (int j = 0; j < nRows; j++) {
			const float* __restrict__ row = Data.GetRowNew(j);
			OPTFLOAT distance = CV.SquaredDistance(i-1, vec, row);
			if (ClosestDistances[j] > distance) {
				ClosestDistances[j] = distance;
			}
		}

		// Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2
		EXPFLOAT sum = 0;
		for (int j = 0; j < nRows; j++) {
			sum += (ClosestDistances[j]*Weights[j]);
			AggregatedSum[j] = sum;
		}

		for(int j=0;j<nRows;j++)
			AggregatedSum[j]/=sum;

		EXPFLOAT r = Rand();
		for (int j = 0; j < Data.GetRowCount(); j++) {
			if (r <= AggregatedSum[j]) {
				for (int l = 0; l < ncols; l++) {
					vec[i * ncols + l] = Data.GetRowNew(j)[l];
				}
				break;
			}
		}
	}

}
