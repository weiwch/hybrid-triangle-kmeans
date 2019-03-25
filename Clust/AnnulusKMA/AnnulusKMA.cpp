#include "AnnulusKMA.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>



AnnulusKMA::AnnulusKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR) :
	AnnulusKMA(aCV,Data,pR, new HamerlyOpenMPSmallestDistances(aCV))
{

}


AnnulusKMA::AnnulusKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,HamerlySmallestDistances *pD) :
		HamerlyKMA(aCV, Data,pR,pD) {
	SecondaryCenter.SetSize(Data.GetRowCount());
	PointNorm.SetSize(Data.GetRowCount());

	CentroidNorms.SetSize(nclusters);
}

void AnnulusKMA::PointAllCtrs(const Array<OPTFLOAT> &vec, const int row_index, const ThreadPrivateVector<OPTFLOAT > &row) {
	OPTFLOAT distance;
	OPTFLOAT smallestDistance = std::numeric_limits<OPTFLOAT>::max();;
	OPTFLOAT secondSmallestDistance = std::numeric_limits<OPTFLOAT>::max();;

	int assignment = -1;
	int secondaryCenter = -1;

	// Update assigned centers
	for (int j = 0; j < nclusters; j++) {
		distance = CV.SquaredDistance(j, vec, row);
		if (distance < smallestDistance) {
			// New closest center found
			secondSmallestDistance = smallestDistance;
			smallestDistance = distance;

			secondaryCenter = assignment;
			assignment = j;
		} else if (distance < secondSmallestDistance) {
			// New second-closest center found
			secondSmallestDistance = distance;
			secondaryCenter = j;
		}
	}

	// Update upper bounds
	UpperBounds[row_index] = std::sqrt(smallestDistance);

	// Update lower bounds
	LowerBounds[row_index] = std::sqrt(secondSmallestDistance);

	// Update assignment
	Assignment[row_index] = assignment;

	// Update secondary center
	SecondaryCenter[row_index] = secondaryCenter;


}

/*
 * Two importantAssumptions (from the analysis of OuterLoop):
 * 		1) UpperBounds holds exact distance to cluster numbered Assignment[row_index];
 * 		2) LowerBounds holds exact Distance to cluster numbered SecondaryCenter[rowindex];
 */
void AnnulusKMA::PointAllCtrs(int lower, int upper, const Array<OPTFLOAT> &vec, const int row_index, const ThreadPrivateVector<OPTFLOAT > &row, long *distanceCount) {
	OPTFLOAT distance;
	OPTFLOAT smallestDistance = UpperBounds[row_index] * UpperBounds[row_index];
	OPTFLOAT secondSmallestDistance = LowerBounds[row_index] * LowerBounds[row_index];

	int assignment = Assignment[row_index];
	int secondaryCenter = SecondaryCenter[row_index];

	// There is such a possibility after the centers move. In such case we must exchange data
	if (smallestDistance>secondSmallestDistance) {
		std::swap(smallestDistance, secondSmallestDistance);
		std::swap(assignment, secondaryCenter);
	}

	int k;
	int count = 0;

	// Update assigned centers
	for (int j = lower; j <= upper; j++) {
		k = CentroidNorms[j].Centroid;

		// But cannot compute distances to the same centers twice.
		// This would be inefficient and incorrectly compute secondSmallestDistance
		if (k==Assignment[row_index] || k==SecondaryCenter[row_index])
			continue;
		distance = CV.SquaredDistance(k, vec, row);
		count++;
		if (distance < smallestDistance) {
			// New closest center found
			secondSmallestDistance = smallestDistance;
			smallestDistance = distance;

			secondaryCenter = assignment;
			assignment = k;
		} else if (distance < secondSmallestDistance) {
			// New second-closest center found
			secondSmallestDistance = distance;
			secondaryCenter = k;
		}
	}

	// Update upper bounds
	UpperBounds[row_index] = std::sqrt(smallestDistance);

	// Update lower bounds
	LowerBounds[row_index] = std::sqrt(secondSmallestDistance);

	// Update assignment
	Assignment[row_index] = assignment;

	// Update secondary center
	SecondaryCenter[row_index] = secondaryCenter;

	// Update distance count
	(*distanceCount) += count;
}

void AnnulusKMA::InitDataStructures(Array<OPTFLOAT> &vec) {
	FillCentroidNorms(vec);
	HamerlyKMA::InitDataStructures(vec);
	const int rowCount = Data.GetRowCount();

#pragma omp parallel for default(none)
	for (int i = 0; i < rowCount; i++) {
		PointNorm[i] = std::sqrt(CV.SquaredRowNorm(Data.GetRowNew(i)));
	}
}

bool AnnulusKMA::CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount) {

	bool cont = false;


	FillCentroidNorms(vec);

	// Sort centers by norm
	SortCenters();

	FillSmallestDistances(vec);
	(*distanceCount) += nclusters * (nclusters - 1) / 2;

	cont = OuterLoop(vec, distanceCount);

	ReduceOMPData();

	ReduceMPIData(pOMPReducer->GetThreadCenter(0), pOMPReducer->GetThreadCounts(0), cont);
	MoveCenters(vec);
	UpdateBounds();

	return cont;
}

double AnnulusKMA::ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount) {
	FillCentroidNorms(vec);

	// Sort centers by norm
	SortCenters();

	FillSmallestDistances(vec);

	OuterLoop(vec, distanceCount);

	ReduceOMPData();

	EXPFLOAT SSE = ComputeSSE(vec);
	ReduceMPIData(pOMPReducer->GetThreadCenter(0), pOMPReducer->GetThreadCounts(0), SSE);
	MoveCenters(vec);
	UpdateBounds();

	return SSE / Data.GetTotalRowCount();
}

void AnnulusKMA::FillCentroidNorms(const Array<OPTFLOAT> &vec) {
	for (int i = 0; i < nclusters; i++) {
		CentroidNorms[i].Centroid = i;
		CentroidNorms[i].Norm = std::sqrt(CV.SquaredCentroidNorm(vec, i));
	}
}

bool compare(const CentroidNorm &a, const CentroidNorm &b) {
	return a.Norm < b.Norm;
}

void AnnulusKMA::SortCenters() {
	// Sort the vector by the norms
	std::sort(CentroidNorms.GetData(), CentroidNorms.GetData()+nclusters, compare);
}

bool AnnulusKMA::OuterLoop(Array<OPTFLOAT> &vec, long *distanceCount) {
	bool cont = false;
	OPTFLOAT m;
	OPTFLOAT r;
	OPTFLOAT distance;
	int lower;
	int upper;
	const int rowCount = Data.GetRowCount();
	long count = 0;

#pragma omp parallel default(none) shared(vec) private(m, r, distance, lower, upper) \
	reduction(+ : count) reduction(|| : cont)
	{
		ResetOMPData();
		int threadId = omp_get_thread_num();
		ThreadPrivateVector<OPTFLOAT> &tpRow = OMPData[threadId].row;

#pragma omp for OMPDYNAMIC
		for (int i = 0; i < rowCount; i++) {
			const float* __restrict__ row = Data.GetRowNew(i);
			m = std::max(SmallestDistances[Assignment[i]] / 2, LowerBounds[i]);

			if (UpperBounds[i] > m) {
				CV.ConvertToOptFloat(tpRow,row);
				UpperBounds[i] = std::sqrt(CV.SquaredDistance(Assignment[i], vec, tpRow));
				count++;
				if (UpperBounds[i] > m) {
					LowerBounds[i] = std::sqrt(CV.SquaredDistance(SecondaryCenter[i], vec, tpRow));
					count++;

					r = std::max(LowerBounds[i], UpperBounds[i]);

					int a = Assignment[i];

					BinarySearch(r, PointNorm[i], lower, upper);
#ifdef _DEBUG
					DbgVerifyRadius(r, PointNorm[i], lower, upper);
#endif
					PointAllCtrs(lower, upper, vec, i, tpRow, &count);
					if (a != Assignment[i]) {
						cont = true;
						MoveObject(a, i, tpRow);
					}
					count += (upper - lower) + 1;
				}
			}
		}
	}

	(*distanceCount) += count;

	return cont;
}

void AnnulusKMA::BinarySearch(OPTFLOAT r, OPTFLOAT pointNorm, int &lower, int &upper) {
	int i;
	int lfirst = 0;
	int llast = nclusters - 1;

	int j;
	int ufirst = 0;
	int ulast = nclusters - 1;

	while (lfirst != llast) {
		i = lfirst + ((llast - lfirst) / 2);

		if ((pointNorm - r) <= CentroidNorms[i].Norm) {
			llast = i;
		} else {
			lfirst = i + 1;
		}
	}

	while (ufirst != ulast) {
		j = ulast - ((ulast - ufirst) / 2);

		if (CentroidNorms[j].Norm <= (pointNorm + r)) {
			ufirst = j;
		} else {
			ulast = j - 1;
		}
	}

	lower = lfirst;
	upper = ufirst;
}

void AnnulusKMA::DbgVerifyRadius(OPTFLOAT r, OPTFLOAT pointNorm, int lower, int upper) {
	for(int i = 0; i < nclusters; i++) {
		if (i >= lower && i <= upper) {
			if (!(std::abs(pointNorm - CentroidNorms[i].Norm) <= r)) {
				printf("Centroid %d is in radius but shouldn't be!\n", i);
			}
		} else {
			if (std::abs(pointNorm - CentroidNorms[i].Norm) <= r) {
				printf("Centroid %d is not in radius but should be!\n", i);
			}
		}
	}
}
