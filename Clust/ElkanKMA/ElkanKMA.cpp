#include "ElkanKMA.h"
#include "../KMeansWithoutMSE/KMeansWithoutMSE.h"
#include "../../Util/OpenMP.h"
#include "../../Util/StdOut.h"

#include <limits>
#include <algorithm>
#include <cmath>

ElkanKMA::ElkanKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,ElkanSmallestDistances *pD) :
		KMeansAlgorithm(aCV, Data,pR),
		KMeansWithoutMSE(aCV, Data, KMeansAlgorithm::IterCount),
		LowerBounds(LargeMatrix<OPTFLOAT>(Data.GetRowCount(),CV.GetNClusters())),
		Distances(Matrix<OPTFLOAT>(CV.GetNClusters(), CV.GetNClusters())) {
	Center.SetSize(nclusters * ncols);
	Counts.SetSize(nclusters);
	pOMPReducer = new Log2OpenMPReducer(aCV);

	SmallestDistances.SetSize(nclusters);
	DistanceMoved.SetSize(nclusters);
	Assignment.SetSize(Data.GetRowCount());
	UpperBounds.SetSize(Data.GetRowCount());
	tmp_vec.SetSize(nclusters * ncols);

	pSmallestDistances=pD;
	OMPData.SetSize(omp_get_max_threads());
#pragma omp parallel
	{
		int n = omp_get_thread_num();
		OMPData[n].Distances.SetSize(nclusters,nclusters);
		OMPData[n].SmallestDistances.SetSize(nclusters);
		OMPData[n].row.SetSize(ncols);
		pOMPReducer->ClearThreadData(n);
	}
}

ElkanKMA::ElkanKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR) :
		ElkanKMA(aCV,Data,pR,new ElkanOpenMPSmallestDistances(aCV)) {

}


void ElkanKMA::InitDataStructures(Array<OPTFLOAT> &vec) {

		const int nRows=Data.GetRowCount();
#pragma omp parallel for default(none) firstprivate(nRows)
	for (int i = 0; i < nRows; i++) {
		Assignment[i] = 0;
		UpperBounds[i] = std::numeric_limits<OPTFLOAT>::max();

		for (int j = 0; j < nclusters; j++) {
			LowerBounds(i, j) = 0;
		}
	}
}

bool ElkanKMA::CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount) {
	bool cont = false;

	ClearDataStructures();

	FillSmallestDistances(vec);
	// add the distances required to compute the SmallestDistances array into the count
	(*distanceCount) += nclusters * (nclusters - 1) / 2;

	cont = OuterLoop(vec, distanceCount);

	ReduceMPIData(Center, Counts, cont);
	MoveCentroids(vec);
	(*distanceCount) += nclusters;

	UpdateBounds();

	return cont;
}

double ElkanKMA::ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount) {
	bool r = true;
	OPTFLOAT z;

	ClearDataStructures();

	FillSmallestDistances(vec);
	// add the distances required to compute the SmallestDistances array into the count
	(*distanceCount) += nclusters * (nclusters - 1) / 2;

	OuterLoop(vec, distanceCount);

	EXPFLOAT SSE = ComputeSSE(vec);
	ReduceMPIData(Center, Counts, SSE);
	MoveCentroids(vec);
	(*distanceCount) += nclusters;

	UpdateBounds();

	return SSE / Data.GetTotalRowCount();
}

bool ElkanKMA::OuterLoop(Array<OPTFLOAT> &vec, long *distanceCount) {
	bool cont = false;
	long count = 0;
	bool r = true;
	OPTFLOAT z;

#pragma omp parallel default(none) private(r, z) shared(vec) \
		reduction(+ : count) reduction(|| : cont)
	{
		const int threadNum = omp_get_thread_num();
		ThreadPrivateVector<OPTFLOAT> &myCenter = pOMPReducer->GetThreadCenter(threadNum);
		ThreadPrivateVector<int> &myCounts = pOMPReducer->GetThreadCounts(threadNum);
		const Matrix<OPTFLOAT> &tpDistances=OMPData[threadNum].Distances;
		const ThreadPrivateVector<OPTFLOAT> &tpSmallestDistances=OMPData[threadNum].SmallestDistances;
		ThreadPrivateVector<OPTFLOAT> &tpRow=OMPData[threadNum].row;
		const int nRows=Data.GetRowCount();
#pragma omp for OMPDYNAMIC
		for (int i = 0; i < nRows; i++) {
			const float *row = Data.GetRowNew(i);
			CV.ConvertToOptFloat(tpRow,row);
			int A = Assignment[i];
			OPTFLOAT U = UpperBounds[i];
			OPTFLOAT * __restrict__ LowerBoundsRow = LowerBounds(i);

			if (U > tpSmallestDistances[A]) {
				r = true;
				for (int j = 0; j < nclusters; j++) {
					OPTFLOAT LB = LowerBoundsRow[j];

					z = std::max(LB, tpDistances(A, j));
					if (j != A && U > z) {
						if (r) {
							U = std::sqrt(CV.SquaredDistance(A, vec, tpRow));
							count++;
							r = false;
						}
						if (U > z) {
							LB = std::sqrt(CV.SquaredDistance(j, vec, tpRow));
							count++;
							if (LB < U) {
								U = LB;
								cont = true;
								A = j;
							}
						}
					}
					LowerBoundsRow[j] = LB;
				}
				UpperBounds[i] = U;
				Assignment[i] = A;
			}


			myCounts[A]++;
			CV.AddRow(A,myCenter,tpRow);
		}
	}

	(*distanceCount) += count;
	pOMPReducer->ReduceToArrays(Center,Counts);

	return cont;
}

void ElkanKMA::UpdateBounds() {

	const int nRows=Data.GetRowCount();
#pragma omp parallel for default(none) firstprivate(nRows)
	for (int i = 0; i < nRows; i++) {
		UpperBounds[i] += DistanceMoved[Assignment[i]];

		OPTFLOAT * __restrict__ LowerBoundsRow = LowerBounds(i);

		for (int j = 0; j < nclusters; j++) {
			LowerBoundsRow[j] -= DistanceMoved[j];
		}
	}
}

void ElkanKMA::ClearDataStructures() {
	int nthreads = omp_get_max_threads();

#pragma omp parallel default(none) firstprivate(nthreads)
	{
		int n = omp_get_thread_num();
		pOMPReducer->ClearThreadData(n);
	}


	for (int j = 0; j < nclusters * ncols; j++) {
		Center[j] = 0;
	}

	for (int j = 0; j < nclusters; j++) {
		Counts[j] = 0;
	}
}

double ElkanKMA::MoveCentroids(Array<OPTFLOAT> &vec) {
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

	return 0;
}

EXPFLOAT ElkanKMA::ComputeSSE(const Array<OPTFLOAT> &vec) {
	EXPFLOAT ret = 0;
	const int rowCount = Data.GetRowCount();

#pragma omp parallel for default(none) shared(vec) reduction(+ : ret)
	for (int i = 0; i < rowCount; i++) {
		ret += CV.SquaredDistance(Assignment[i], vec, Data.GetRowNew(i));
	}

	return ret;
}

void ElkanKMA::FillSmallestDistances(const Array<OPTFLOAT> &vec) {
	pSmallestDistances->FillSmallestDistances(vec,Distances,SmallestDistances);
#pragma omp parallel
	{
		int n=omp_get_thread_num();
		OMPData[n].Distances=Distances;
		ThreadPrivateVector<OPTFLOAT> &tpSD=OMPData[n].SmallestDistances;
		for(int i=0;i<nclusters;i++)
			tpSD[i]=SmallestDistances[i];
	}
}

void ElkanKMA::PrintNumaLocalityInfo() {
	double assignmentLoc = NumaLocalityofArray(Assignment) * 100.0;
	double upperBoundsLoc = NumaLocalityofArray(UpperBounds) * 100.0;
	double lowerBoundsLoc = NumaLocalityofMatrix(LowerBounds) * 100.0;
	double distanceMovedLoc = NumaLocalityofPrivateArray(DistanceMoved) * 100.0;
	double tmpvecLoc = NumaLocalityofPrivateArray(tmp_vec) * 100.0;
	double distancesLoc = NumaLocalityofPrivateMatrix(Distances)* 100.0;
	double smallestDistancesLoc=NumaLocalityofPrivateArray(SmallestDistances) * 100.0;
	double centerLoc=NumaLocalityofPrivateArray(Center) * 100.0;
	double countsLoc=NumaLocalityofPrivateArray(Counts) * 100.0;

	kma_printf("NUMA locality of Assignment is %1.2f%%\n", assignmentLoc);
	kma_printf("NUMA locality of UpperBounds is %1.2f%%\n", upperBoundsLoc);
	kma_printf("NUMA locality of LowerBounds is %1.2f%%\n", lowerBoundsLoc);

	kma_printf("NUMA locality of thread 0 array Center is %1.2f%%\n", centerLoc);
	kma_printf("NUMA locality of thread 0 array Counts is %1.2f%%\n", countsLoc);
	kma_printf("NUMA locality of thread 0 array DistanceMoved is %1.2f%%\n", distanceMovedLoc);
	kma_printf("NUMA locality of thread 0 array tmp_vec is %1.2f%%\n", tmpvecLoc);
	kma_printf("NUMA locality of thread 0 array SmallestDistances is %1.2f%%\n", smallestDistancesLoc);
	kma_printf("NUMA locality of thread 0 matrix Distances is %1.2f%%\n", distancesLoc);


#pragma omp parallel default(none)
	{
		int id = omp_get_thread_num();
		ThreadPrivateVector<OPTFLOAT> &myCenter = pOMPReducer->GetThreadCenter(id);
		ThreadPrivateVector<int> &myCounts = pOMPReducer->GetThreadCounts(id);

		double centerLoc = NumaLocalityofPrivateArray(myCenter) * 100.0;
		double countLoc = NumaLocalityofPrivateArray(myCounts) * 100.0;
		double tpdistancesLoc=NumaLocalityofPrivateMatrix(OMPData[id].Distances) *100.0;
		double tpsmallestDistancesLoc=NumaLocalityofPrivateArray(OMPData[id].SmallestDistances) * 100.0;

		kma_printf("Thread %d NUMA locality of private array OMPData[%d].Center is %1.2f%%\n", id, id, centerLoc);
		kma_printf("Thread %d NUMA locality of private array OMPData[%d].Counts is %1.2f%%\n", id, id, countLoc);
		kma_printf("Thread %d NUMA locality of private matrix OMPData[%d].Distances is %1.2f%%\n", id, id, tpdistancesLoc);
		kma_printf("Thread %d NUMA locality of private array OMPData[%d].SmallestDistances is %1.2f%%\n", id, id, tpsmallestDistancesLoc);
	}

}

ElkanKMA::~ElkanKMA() {
	delete pOMPReducer;
	delete pSmallestDistances;
}
