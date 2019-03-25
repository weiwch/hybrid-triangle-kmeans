#ifndef DrakeKMA_H_
#define DrakeKMA_H_

#include "../KMAlgorithm.h"
#include "../KMeansWithoutMSE/KMeansWithoutMSE.h"
#include "../../Util/LargeVector.h"
#include "../../Util/OpenMP.h"

struct CentroidPointDistance {
	OPTFLOAT Distance;
	int Centroid;
	int OryginalPosition;
};

struct DrakeThreadData {
	DynamicArray<CentroidPointDistance> CentroidPointDistances;
	ThreadPrivateVector<OPTFLOAT> row;
}__attribute__ ((aligned (64)));

class DrakeKMA: public KMeansAlgorithm, public KMeansWithoutMSE {

protected:

	/**
	 * Number of lower bounds "b": 1 < b < k - 1.
	 */
	int B;

	/**
	 * Use adaptive Drake bounds?
	 */
	bool AdaptiveDrake;

	/**
	 * Is it the first KMA run?
	 */
	bool FirstRun;

	/**
	 * Defines to which center the data row (datapoint) [index] is currently assigned.
	 */
	LargeMatrix<int> BoundsAssignment;

	/**
	 *
	 */
	LargeVector<int> Assignment;

	/**
	 * Per thread data.
	 */
	DynamicArray<DrakeThreadData> OMPData;

	/**
	 * Upper bound on the distance between data row [index] and
	 * its assigned center centroidis[UpperBounds[index]].
	 */
	LargeVector<OPTFLOAT> UpperBounds;

	/**
	 * Lower bound on the distance between the data row [index] and
	 * its second closest center i.e. the closest center to the data row
	 * that is not centroids[Assignment[i]].
	 */
	LargeMatrix<OPTFLOAT> LowerBounds;

	/**
	 * The distance the [index] centroid last moved.
	 */
	DynamicArray<OPTFLOAT> DistanceMoved;

	/**
	 * Vector of the sum through all the objects
	 * belonging to the given centroid. Represented as:
	 * [1-st centroid sum, 2-nd centroid sum, ...]
	 * The centroids are calculated as Centers/Counts.
	 */
	ThreadPrivateVector<OPTFLOAT> Center;

	/**
	 * The number of objects belonging to the given centroid.
	 * The centroids are calculated as Center/Counts.
	 */
	ThreadPrivateVector<int> Counts;

	/**
	 * The temporary centroid vector.
	 */
	DynamicArray<OPTFLOAT> tmp_vec;

	/**
	 * The OpenMP reducer.
	 */
	OpenMPKMAReducer *pOMPReducer;

	/**
	 * Calculate B for adaptive Drake.
	 */
	void calculateB();

	/**
	 * Clear the structures used by the algorithm.
	 */
	void ClearDataStructures();

	/**
	 * Sort-Centers from Drake KMA.
	 */
	void SortCenters(const int row_index, const int FirstBound, const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row);
	void SortAllCenters(const int row_index,const Array<OPTFLOAT> &vec,const ThreadPrivateVector<OPTFLOAT> &row);

	/**
	 * Init auxiliary data structures used by the algorithm
	 * given initial centroid coordinates.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	virtual void InitDataStructures(Array<OPTFLOAT> &vec);

	/**
	 * Move the centroids to their new positions and calculate the Distance
	 * they moved.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	void MoveCenters(Array<OPTFLOAT> &vec);

	/**
	 * Compute new centers by filling the vec array with the result of
	 * Centers/Count.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	void ComputeNewCenters(Array<OPTFLOAT> &vec);

	/**
	 * The outer DrakeKMA loop shared between different versions.
	 */
	virtual bool OuterLoop(Array<OPTFLOAT> &vec, long *distanceCount);

	/**
	 * Update-Bounds function from Making K-means even faster.
	 */
	void UpdateBounds();

	/**
	 * Reduce OMPData (DeltaCenter, DeltaCounts) to Center and Counts.
	 */
	inline void ReduceOMPData() {
		pOMPReducer->ReduceToZero();
	}

	/**
	 * Clear the OMPData array.
	 */
	inline void ResetOMPData() {
		int threadId = omp_get_thread_num();
		pOMPReducer->ClearThreadData(threadId);
	}

	/**
	 * Reduce MPI data between processes.
	 */
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &SSE, int &MaxB) {}
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bcont, int &MaxB) {}
	virtual void PrintIterInfo(int i, double BestFit, double Rel,double Avoided, double iterTime);

public:
	DrakeKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR, int b, bool AdaptiveDrake);
	~DrakeKMA();

	virtual double ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount);

	bool CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount);

	void RunKMeansWithoutMSE(Array<OPTFLOAT> &vec, const int verbosity);

	void PrintNumaLocalityInfo();

	/**
	 * Compute the SSE for the given centroid vector and dataset.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	EXPFLOAT ComputeSSE(const Array<OPTFLOAT> &vec);
};

#endif /*DrakeKMA_H_*/
