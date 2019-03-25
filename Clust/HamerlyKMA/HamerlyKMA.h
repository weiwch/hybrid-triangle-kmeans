#ifndef HamerlyKMA_H_
#define HamerlyKMA_H_

#include "../KMAlgorithm.h"
#include "../KMeansWithoutMSE/KMeansWithoutMSE.h"
#include "HamerlySmallestDistances.h"
#include "../../Util/LargeVector.h"
#include "../../Util/OpenMP.h"

struct HamerlyThreadData {
	ThreadPrivateVector<OPTFLOAT> SmallestDistances;
	ThreadPrivateVector<OPTFLOAT> row;
}__attribute__ ((aligned (64)));





class HamerlyKMA: public KMeansAlgorithm, public KMeansWithoutMSE {

protected:

	HamerlySmallestDistances *pSmallestDistances;

	/**
	 * Distance from center [index] to its closest other center.
	 */
	ThreadPrivateVector<OPTFLOAT> SmallestDistances;

	/**
	 * Defines to which center the data row (datapoint) [index] is currently assigned.
	 */
	LargeVector<int> Assignment;

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
	LargeVector<OPTFLOAT> LowerBounds;


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


	DynamicArray<HamerlyThreadData> OMPData;

	/**
	 * The OpenMP reducer.
	 */
	OpenMPKMAReducer *pOMPReducer;

	/**
	 * Move the given object from cluster to cluster.
	 *  previousAssignment - index of the previously assigned cluster
	 *  i - object index
	 *  row - the data row to move
	 */
	void MoveObject(const int previousAssignment, const int i, const ThreadPrivateVector<OPTFLOAT > &row);

	/**
	 * Fill the SmallestDistances array.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	void FillSmallestDistances(const Array<OPTFLOAT> &vec);

	/**
	 * Init auxiliary data structures used by the algorithm
	 * given initial centroid coordinates.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	virtual void InitDataStructures(Array<OPTFLOAT> &vec);

	/**
	 * The Point-All-Ctrs function from Making K-means even faster.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 *  row_index - the index of the row in the Dataset
	 *  row - the data row as a float array
	 */
	virtual void PointAllCtrs(const Array<OPTFLOAT> &vec, const int row_index, const ThreadPrivateVector<OPTFLOAT > &row);

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
	 * The outer HamerlyKMA loop shared between different versions.
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
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &SSE) {}
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bcont) {}

public:
	HamerlyKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR);
	HamerlyKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,HamerlySmallestDistances *pD);

	~HamerlyKMA();

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

#endif /*HamerlyKMA_H_*/
