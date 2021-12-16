#include "HamerlyKMA.h"

#include <limits>
#include <algorithm>
#include <cmath>

#include "../../Util/OpenMP.h"
#include "../../Util/StdOut.h"

HamerlyKMA::HamerlyKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,HamerlySmallestDistances *pD) :
		KMeansAlgorithm(aCV, Data,pR),
		KMeansWithoutMSE(aCV, Data, IterCount) {
	Center.SetSize(nclusters * ncols);
	Counts.SetSize(nclusters);

	pOMPReducer = new Log2OpenMPReducer(aCV);

	DistanceMoved.SetSize(nclusters);
	Assignment.SetSize(Data.GetRowCount());
	UpperBounds.SetSize(Data.GetRowCount());
	LowerBounds.SetSize(Data.GetRowCount());
	tmp_vec.SetSize(nclusters * ncols);

	SmallestDistances.SetSize(nclusters);
	pSmallestDistances=pD;
	OMPData.SetSize(omp_get_max_threads());

#pragma omp parallel
	{
		int n = omp_get_thread_num();
		pOMPReducer->ClearThreadData(n);
		OMPData[n].SmallestDistances.SetSize(nclusters);
		OMPData[n].row.SetSize(ncols);
	}
}

HamerlyKMA::HamerlyKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR) :
		HamerlyKMA(aCV,Data,pR, new HamerlyOpenMPSmallestDistances(aCV)) {

}


HamerlyKMA::~HamerlyKMA() {
	delete pOMPReducer;
	delete pSmallestDistances;
}

bool HamerlyKMA::CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount) {

	bool cont = false;

	FillSmallestDistances(vec);
	(*distanceCount) += nclusters * (nclusters - 1) / 2;

	cont = OuterLoop(vec, distanceCount);

	ReduceOMPData();

	ReduceMPIData(pOMPReducer->GetThreadCenter(0), pOMPReducer->GetThreadCounts(0), cont);
	MoveCenters(vec);
	UpdateBounds();

	return cont;
}

double HamerlyKMA::ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount) {

	FillSmallestDistances(vec);
	(*distanceCount) += nclusters * (nclusters - 1) / 2;

	OuterLoop(vec, distanceCount);

	ReduceOMPData();

	EXPFLOAT SSE = ComputeSSE(vec);
	ReduceMPIData(pOMPReducer->GetThreadCenter(0), pOMPReducer->GetThreadCounts(0), SSE);
	MoveCenters(vec);
	UpdateBounds();

	return SSE / Data.GetTotalRowCount();
}

bool HamerlyKMA::OuterLoop(Array<OPTFLOAT> &vec, long *distanceCount) {
	bool cont = false;
	OPTFLOAT m;
	int a;
	int assignment;
	const int rowCount = Data.GetRowCount();
	long count = 0;

#pragma omp parallel default(none) shared(vec) private(assignment, m, a) firstprivate(rowCount) \
	reduction(+ : count) reduction(|| : cont)
	{
		ResetOMPData();
		int threadId = omp_get_thread_num();
		ThreadPrivateVector<OPTFLOAT> &tpSmallestDistances=OMPData[threadId].SmallestDistances;
		ThreadPrivateVector<OPTFLOAT> &tpRow = OMPData[threadId].row;

		tpSmallestDistances=SmallestDistances;


#pragma omp for OMPDYNAMIC
		for (int i = 0; i < rowCount; i++) {
			const float* __restrict__ row = Data.GetRowNew(i);
			assignment = Assignment[i];
			m = std::max(tpSmallestDistances[assignment] / 2, LowerBounds[i]);
			if (UpperBounds[i] > m) {
				CV.ConvertToOptFloat(tpRow,row);
				UpperBounds[i] = std::sqrt(
						CV.SquaredDistance(assignment, vec, tpRow));

				if (UpperBounds[i] > m) {
					a = assignment;
					PointAllCtrs(vec, i, tpRow);
					if (a != Assignment[i]) {
						MoveObject(a, i, tpRow);
						cont = true;
					}
					count++;
				}
			}
		}
	}

	(*distanceCount) += count * nclusters;

	return cont;
}

void HamerlyKMA::MoveObject(const int previousAssignment, const int i, const ThreadPrivateVector<OPTFLOAT > &row) {
	const int newAssignment = Assignment[i];

	int threadId = omp_get_thread_num();
	ThreadPrivateVector<OPTFLOAT> &DeltaCenter = pOMPReducer->GetThreadCenter(threadId);
	ThreadPrivateVector<int> &DeltaCounts = pOMPReducer->GetThreadCounts(threadId);

	CV.MoveRow(previousAssignment,newAssignment,DeltaCenter,row);

	DeltaCounts[previousAssignment]--;
	DeltaCounts[newAssignment]++;
}

void HamerlyKMA::FillSmallestDistances(const Array<OPTFLOAT> &vec) {
	pSmallestDistances->FillSmallestDistances(vec,SmallestDistances);
}

void HamerlyKMA::InitDataStructures(Array<OPTFLOAT> &vec) {

	// Zeroing
	// TODO: Is it really necessary?
	for (int i = 0; i < nclusters; i++) {
		Counts[i] = (OPTFLOAT) 0.0;
	}

	for (int i = 0; i < nclusters * ncols; i++) {
		Center[i] = (OPTFLOAT) 0.0;
	}

	const int rowCount = Data.GetRowCount();
#pragma omp parallel default(none) shared(vec) firstprivate(rowCount)
	{
		int threadId = omp_get_thread_num();
		ThreadPrivateVector<int> &DeltaCounts = pOMPReducer->GetThreadCounts(threadId);
		ThreadPrivateVector<OPTFLOAT> &DeltaCenter = pOMPReducer->GetThreadCenter(threadId);
		ThreadPrivateVector<OPTFLOAT> &tpRow = OMPData[threadId].row;
#pragma omp for
		for (int i = 0; i < rowCount; i++) {
			const float *row = Data.GetRowNew(i);

			CV.ConvertToOptFloat(tpRow,row);
			PointAllCtrs(vec, i, tpRow);
			DeltaCounts[Assignment[i]]++;

			for (int j = 0; j < ncols; j++) {
				// TODO: Check if ok.
				DeltaCenter[Assignment[i] * ncols + j] += tpRow[j];
			}
		}
	}
	ReduceOMPData();
	bool bCont=true;
	ReduceMPIData(pOMPReducer->GetThreadCenter(0), pOMPReducer->GetThreadCounts(0), bCont);
	ThreadPrivateVector<OPTFLOAT> &zeroCenter = pOMPReducer->GetThreadCenter(0);
	ThreadPrivateVector<int> &zeroCounts = pOMPReducer->GetThreadCounts(0);
	pOMPReducer->AddZeroToCenter(Center, Counts);

}

EXPFLOAT HamerlyKMA::ComputeSSE(const Array<OPTFLOAT> &vec) {
	EXPFLOAT ret = 0;
	const int rowCount = Data.GetRowCount();

#pragma omp parallel for default(none) shared(vec) firstprivate(rowCount) reduction(+ : ret)
	for (int i = 0; i < rowCount; i++) {
		ret += CV.SquaredDistance(Assignment[i], vec, Data.GetRowNew(i));
	}

	return ret;
}

void HamerlyKMA::PointAllCtrs(const Array<OPTFLOAT> &vec, const int row_index, const ThreadPrivateVector<OPTFLOAT > &row) {
	OPTFLOAT distance;
	OPTFLOAT smallestDistance = std::numeric_limits<OPTFLOAT>::max();
	OPTFLOAT secondSmallestDistance = std::numeric_limits<OPTFLOAT>::max();

	int assignment = -1;

	// Update assigned centers
	for (int j = 0; j < nclusters; j++) {
		distance = CV.SquaredDistance(j, vec, row);
		if (distance < smallestDistance) {
			secondSmallestDistance = smallestDistance;
			smallestDistance = distance;
			assignment = j;
		} else if(distance < secondSmallestDistance) {
			secondSmallestDistance = distance;
		}
	}
	Assignment[row_index] = assignment;

	// Update upper bounds
	UpperBounds[row_index] = std::sqrt(smallestDistance);

	// Update lower bounds
	LowerBounds[row_index] = std::sqrt(secondSmallestDistance);

}

void HamerlyKMA::MoveCenters(Array<OPTFLOAT> &vec) {
	// TODO: Memcopy, parallel or simd
	for (int i = 0; i < nclusters * ncols; i++) {
		tmp_vec[i] = vec[i];
	}

	int centerSize = nclusters * ncols;
	ThreadPrivateVector<OPTFLOAT> &zeroCenter = pOMPReducer->GetThreadCenter(0);
	ThreadPrivateVector<int> &zeroCounts = pOMPReducer->GetThreadCounts(0);

	pOMPReducer->AddZeroToCenter(Center, Counts);

	ComputeNewCenters(vec);

	for (int i = 0; i < nclusters; i++) {
		DistanceMoved[i] = std::sqrt(CV.SquaredDistance(tmp_vec, vec, i));
	}
}

void HamerlyKMA::ComputeNewCenters(Array<OPTFLOAT> &vec) {
	// TODO: OPTFLOAT or double?
	OPTFLOAT f;

	for (int i = 0; i < nclusters; i++) {
		if (Counts[i] > 0) {
			// TODO: 1.0 should be cast to OPTFLOAT?
			f = 1.0 / Counts[i];
			for (int j = ncols * i; j < ncols * (i + 1); j++) {
				vec[j] = Center[j] * f;
			}
		} else {
			pRepair->RepairVec(vec,i);
		}
	}
}

void HamerlyKMA::UpdateBounds() {
	int r = -1;
	int rprim = -1;
	OPTFLOAT tmp = -std::numeric_limits<OPTFLOAT>::infinity();

	for (int j = 0; j < nclusters; j++) {
		if (tmp < DistanceMoved[j]) {
			tmp = DistanceMoved[j];
			r = j;
		}
	}

	tmp = -std::numeric_limits<OPTFLOAT>::infinity();
	for (int j = 0; j < nclusters; j++) {
		if (j != r) {
			if (tmp < DistanceMoved[j]) {
				tmp = DistanceMoved[j];
				rprim = j;
			}
		}
	}

	const int rowCount = Data.GetRowCount();
#pragma omp parallel for  firstprivate(r, rprim)
	for (int i = 0; i < rowCount; i++) {
		UpperBounds[i] += DistanceMoved[Assignment[i]];
		if (r == Assignment[i]) {
			LowerBounds[i] -= DistanceMoved[rprim];
		} else {
			LowerBounds[i] -= DistanceMoved[r];
		}
	}

}

void HamerlyKMA::PrintNumaLocalityInfo() {
	double assignmentLoc = NumaLocalityofArray(Assignment) * 100.0;
	double upperBoundsLoc = NumaLocalityofArray(UpperBounds) * 100.0;
	double lowerBoundsLoc = NumaLocalityofArray(LowerBounds) * 100.0;
	double distanceMovedLoc = NumaLocalityofArray(DistanceMoved) * 100.0;
	kma_printf("NUMA locality of Assignment is %1.2f%%\n", assignmentLoc);
	kma_printf("NUMA locality of UpperBounds is %1.2f%%\n", upperBoundsLoc);
	kma_printf("NUMA locality of LowerBounds is %1.2f%%\n", lowerBoundsLoc);
	kma_printf("NUMA locality of DistanceMoved is %1.2f%%\n", distanceMovedLoc);
}
