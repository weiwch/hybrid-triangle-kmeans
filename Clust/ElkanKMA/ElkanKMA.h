#ifndef ElkanKMA_H_
#define ElkanKMA_H_

#include "../KMAlgorithm.h"
#include "../KMeansWithoutMSE/KMeansWithoutMSE.h"
#include "ElkanSmallestDistances.h"
#include "../../Util/LargeVector.h"


struct ElkanThreadData {
	Matrix<OPTFLOAT> Distances;
	ThreadPrivateVector<OPTFLOAT> SmallestDistances;
	ThreadPrivateVector<OPTFLOAT> row;

}__attribute__ ((aligned (64)));




class ElkanKMA: public KMeansAlgorithm, public KMeansWithoutMSE {

protected:
	ElkanSmallestDistances *pSmallestDistances;


	/**
	 * Distance from [index] to its closest other center.
	 */
	DynamicArray<OPTFLOAT> SmallestDistances;

	/**
	 * Distance from cluster [index0] to cluster [index1].
	 */
	Matrix<OPTFLOAT> Distances;

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
	LargeMatrix<OPTFLOAT> LowerBounds;

	/**
	 * The distance the [index] centroid last moved.
	 */
	DynamicArray<OPTFLOAT> DistanceMoved;

	/**
	 * The temporary centroid vector.
	 */
	DynamicArray<OPTFLOAT> tmp_vec;

	/**
	 * The array containing the centers and the counts assigned
	 * to the given thread.
	 */

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

	DynamicArray<ElkanThreadData> OMPData;

	/**
	 * The OpenMP reducer.
	 */
	OpenMPKMAReducer *pOMPReducer;


	/**
	 * Fills the SmallestDistances array.
	 */
	void FillSmallestDistances(const Array<OPTFLOAT> &vec);

	/**
	 * Init auxiliary data structures used by the algorithm
	 * given initial centroid coordinates.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	virtual void InitDataStructures(Array<OPTFLOAT> &vec);

	/**
	 * Compute the mean of points for a given centroid and place them in tmp_vec.
	 */
	double MoveCentroids(Array<OPTFLOAT> &vec);

	/**
	 * Update-Bounds function updates Upper and Lower bounds after center recalculation.
	 */
	void UpdateBounds();

	/**
	 * Clear the required data structures for this iteration.
	 */
	void ClearDataStructures();

	/**
	 * The outer ElkanKMA loop shared between different versions.
	 */
	bool OuterLoop(Array<OPTFLOAT> &vec, long *distanceCount);

	/**
	 * Reduce MPI data between processes.
	 */
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &SSE) {}
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bcont) {}

public:
	ElkanKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,ElkanSmallestDistances *pD);
	ElkanKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR);
	~ElkanKMA();

	virtual double ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *innerLoopCount);

	bool CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount);

	virtual void PrintNumaLocalityInfo();

	/**
	 * Compute the SSE for the given centroid vector and dataset.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	EXPFLOAT ComputeSSE(const Array<OPTFLOAT> &vec);

};

#endif /*ElkanKMA_H_*/
