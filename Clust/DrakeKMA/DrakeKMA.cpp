#include "DrakeKMA.h"

#include <limits>
#include <algorithm>
#include <cmath>

#include "../../Util/OpenMP.h"
#include "../../Util/StdOut.h"

DrakeKMA::DrakeKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR, int b, bool adaptiveDrake) :
		KMeansAlgorithm(aCV, Data, pR), KMeansWithoutMSE(aCV, Data, IterCount), B(b), AdaptiveDrake(adaptiveDrake), FirstRun(true) {
	Center.SetSize(nclusters * ncols);
	Counts.SetSize(nclusters);

	pOMPReducer = new Log2OpenMPReducer(aCV);

	DistanceMoved.SetSize(nclusters);
	UpperBounds.SetSize(Data.GetRowCount());
	Assignment.SetSize(Data.GetRowCount());
	tmp_vec.SetSize(nclusters * ncols);
	OMPData.SetSize(omp_get_max_threads());

	calculateB();

	LowerBounds.SetSize(Data.GetRowCount(), B);
	BoundsAssignment.SetSize(Data.GetRowCount(), B);

#pragma omp parallel default(none)
	{
		int threadNum = omp_get_thread_num();
		OMPData[threadNum].CentroidPointDistances.SetSize(nclusters);
		OMPData[threadNum].row.SetSize(ncols);
	}

}

void DrakeKMA::PrintIterInfo(int i, double BestFit, double Rel,double distanceCalculationsRatio, double iterTime) {
	kma_printf("i: %d fit: %g rel: %g, dcr: %g, B: %d itime: %g seconds\n",i,BestFit,Rel,distanceCalculationsRatio,B, iterTime);
}


bool DrakeKMA::CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount) {

	bool cont = false;

	//(*distanceCount) += B * 2 * Data.GetRowCount();

	ClearDataStructures();

	cont = OuterLoop(vec, distanceCount);

	ReduceOMPData();

	ReduceMPIData(Center, Counts, cont, B);

	MoveCenters(vec);
	UpdateBounds();

	return cont;
}

double DrakeKMA::ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount) {

	//(*distanceCount) += B * 2 * Data.GetRowCount();

	ClearDataStructures();

	OuterLoop(vec, distanceCount);

	ReduceOMPData();

	EXPFLOAT SSE = ComputeSSE(vec);
	ReduceMPIData(Center, Counts, SSE, B);

	MoveCenters(vec);
	UpdateBounds();

	return SSE / Data.GetTotalRowCount();
}

void DrakeKMA::ClearDataStructures() {

#pragma omp parallel default(none)
	{
		ResetOMPData();
	}

	for (int j = 0; j < nclusters * ncols; j++) {
		Center[j] = 0;
	}

	for (int j = 0; j < nclusters; j++) {
		Counts[j] = 0;
	}
}


bool DrakeKMA::OuterLoop(Array<OPTFLOAT> &vec, long *distanceCount) {
	bool cont = false;
	OPTFLOAT m;
	int assignment;
	const int rowCount = Data.GetRowCount();
	long count = 0;
	int maxActiveB = 1;

#pragma omp parallel default(none) shared(vec) private(assignment, m) \
	reduction(+ : count) reduction(|| : cont) firstprivate(rowCount) reduction(max: maxActiveB)
	{
		const int threadNum = omp_get_thread_num();
		ThreadPrivateVector<OPTFLOAT> &myCenter = pOMPReducer->GetThreadCenter(threadNum);
		ThreadPrivateVector<int> &myCounts = pOMPReducer->GetThreadCounts(threadNum);
		ThreadPrivateVector<OPTFLOAT> &tpRow=OMPData[threadNum].row;

		ResetOMPData();

#pragma omp for OMPDYNAMIC
		for (int i = 0; i < rowCount; i++) {
			CV.ConvertToOptFloat(tpRow,Data.GetRowNew(i));

			bool sortAll = true;
			int A = Assignment[i];

			for (int bound = 0; sortAll && bound < B; bound++) {
				if (UpperBounds[i] <= LowerBounds(i, bound)) {
					if (bound + 1 > maxActiveB) {
						maxActiveB = bound + 1;
					}
					sortAll = false;
					// If first bound succeeds, assignment stays the same
					if (bound > 0) {
						SortCenters(i, bound, vec,tpRow);
						count += (bound + 1);
					}
				}

			}

			// All bounds failed so we need to recalculate all
			if (sortAll) {
				SortAllCenters(i, vec,tpRow);
				maxActiveB = B;
				count += nclusters;
			}

			if (A != Assignment[i]) {
				cont = true;
			}

			// Drake paper does not mention incremental updates
			myCounts[Assignment[i]]++;
			CV.AddRow(Assignment[i],myCenter,tpRow);

		}
	}

	if(AdaptiveDrake && !FirstRun) {
		B = std::max((int) (ceil((float) nclusters / 8) - 1), maxActiveB);
		B = std::max(B, 1);
	}

	*distanceCount += count;
	pOMPReducer->ReduceToArrays(Center,Counts);

	FirstRun = false;

	return cont;
}

void DrakeKMA::calculateB() {
	if (AdaptiveDrake) {
		B = std::max((int) (ceil(nclusters / (float) (4))), 1);
	}
}

void DrakeKMA::InitDataStructures(Array<OPTFLOAT> &vec) {

	calculateB();

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
		ThreadPrivateVector<OPTFLOAT> &tpRow=OMPData[omp_get_thread_num()].row;
#pragma omp for
		for (int i = 0; i < rowCount; i++) {
			CV.ConvertToOptFloat(tpRow,Data.GetRowNew(i));
			SortAllCenters(i, vec,tpRow);
		}
	}
}

EXPFLOAT DrakeKMA::ComputeSSE(const Array<OPTFLOAT> &vec) {
	EXPFLOAT ret = 0;
	const int rowCount = Data.GetRowCount();

#pragma omp parallel for default(none) shared(vec) firstprivate(rowCount) reduction(+ : ret)
	for (int i = 0; i < rowCount; i++) {
		ret += CV.SquaredDistance(Assignment[i], vec, Data.GetRowNew(i));
	}

	return ret;
}

void DrakeKMA::MoveCenters(Array<OPTFLOAT> &vec) {
	OPTFLOAT f;
	for (int i = 0; i < nclusters; i++) {
		if (Counts[i] > 0) {
			// TODO: 1.0 should be cast to OPTFLOAT?
			f = 1.0 / Counts[i];
			for (int j = ncols * i; j < ncols * (i + 1); j++) {
				tmp_vec[j] = Center[j] * f;
			}
		} else {
			pRepair->RepairVec(tmp_vec,i);
		}
	}

	for (int i = 0; i < nclusters; i++) {
		DistanceMoved[i] = std::sqrt(CV.SquaredDistance(tmp_vec, vec, i));
	}

	for (int i = 0; i < (nclusters * ncols); i++) {
		vec[i] = tmp_vec[i];
	}

	return;
}

void DrakeKMA::UpdateBounds() {
	int m = -1;
	OPTFLOAT tmp = -std::numeric_limits<OPTFLOAT>::infinity();

	for (int j = 0; j < nclusters; j++) {
		if (tmp < DistanceMoved[j]) {
			tmp = DistanceMoved[j];
			m = j;
		}
	}

        const int rowCount = Data.GetRowCount();

#pragma omp parallel for default(none) firstprivate(tmp) firstprivate(rowCount)
	for (int i = 0; i < rowCount; i++) {
		UpperBounds[i] += DistanceMoved[Assignment[i]];
		LowerBounds(i, B - 1) -= tmp;
		for (int z = B - 2; z >= 0; z--) {
			LowerBounds(i, z) -= DistanceMoved[BoundsAssignment(i, z)];
			if (LowerBounds(i, z) > LowerBounds(i, z + 1)) {
				LowerBounds(i)[z] = LowerBounds(i, z + 1);
			}
		}
	}

}

bool compare(const CentroidPointDistance &a, const CentroidPointDistance &b) {
	if (a.Distance < b.Distance) {
		return true;
	} else if (b.Distance < a.Distance) {
		return false;
	} else {
		return a.OryginalPosition < b.OryginalPosition;
	}
}

// firstBound is the number of the first bound, which succeeds
void DrakeKMA::SortCenters(const int row_index, const int firstBound, const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row) {
	ASSERT(firstBound > 0 && firstBound < B);

	int threadNum = omp_get_thread_num();
	DynamicArray<CentroidPointDistance> &cpd = OMPData[threadNum].CentroidPointDistances;


	// First copy the closest center.
	cpd[0].Centroid = Assignment[row_index];
	cpd[0].Distance = std::sqrt(CV.SquaredDistance(Assignment[row_index], vec, row));
	cpd[0].OryginalPosition = 0;

	//Next copy all bounds up to (but not including) firstBound
	for (int k = 0; k < firstBound; k++) {
		cpd[k + 1].Centroid = BoundsAssignment(row_index,k);
		cpd[k + 1].Distance = std::sqrt(CV.SquaredDistance(BoundsAssignment(row_index, k), vec, row));
		cpd[k + 1].OryginalPosition = k;
	}

	std::sort(cpd.GetData(), cpd.GetData() + firstBound + 1, compare);

	Assignment[row_index] = cpd[0].Centroid;
	UpperBounds[row_index] = cpd[0].Distance;
	for (int i = 0; i < firstBound; i++) {
		BoundsAssignment(row_index, i) = cpd[i + 1].Centroid;
		LowerBounds(row_index, i) = cpd[i + 1].Distance;
	}

}

// Sort all centers take first as UpperBound and B next as LowerBounds
void DrakeKMA::SortAllCenters(const int row_index, const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row) {

	int threadNum = omp_get_thread_num();
	DynamicArray<CentroidPointDistance> &cpd = OMPData[threadNum].CentroidPointDistances;

	for (int k = 0; k < nclusters; k++) {
		cpd[k].Centroid = k;
		cpd[k].Distance = std::sqrt(CV.SquaredDistance(k, vec, row));
		cpd[k].OryginalPosition = k;
	}

	std::partial_sort(cpd.GetData(), cpd.GetData() + B + 1, cpd.GetData() + nclusters, compare);

	Assignment[row_index] = cpd[0].Centroid;
	UpperBounds[row_index] = cpd[0].Distance;
	for (int i = 0; i < B; i++) {
		BoundsAssignment(row_index, i) = cpd[i + 1].Centroid;
		LowerBounds(row_index, i) = cpd[i + 1].Distance;
	}

}

void DrakeKMA::PrintNumaLocalityInfo() {
	double assignmentLoc = NumaLocalityofMatrix(BoundsAssignment) * 100.0;
	double upperBoundsLoc = NumaLocalityofArray(UpperBounds) * 100.0;
	double lowerBoundsLoc = NumaLocalityofMatrix(LowerBounds) * 100.0;
	double distanceMovedLoc = NumaLocalityofArray(DistanceMoved) * 100.0;
	kma_printf("NUMA locality of Assignment is %1.2f%%\n", assignmentLoc);
	kma_printf("NUMA locality of UpperBounds is %1.2f%%\n", upperBoundsLoc);
	kma_printf("NUMA locality of LowerBounds is %1.2f%%\n", lowerBoundsLoc);
	kma_printf("NUMA locality of DistanceMoved is %1.2f%%\n", distanceMovedLoc);
}

DrakeKMA::~DrakeKMA() {
	delete pOMPReducer;
}
