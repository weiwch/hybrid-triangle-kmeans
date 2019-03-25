#ifndef ANNULUSKMA_H_
#define ANNULUSKMA_H_

#include <cmath>
#include "../HamerlyKMA/HamerlyKMA.h"

struct CentroidNorm {
	OPTFLOAT Norm;
	int Centroid;

	bool operator<(const CentroidNorm &a) {
		return Norm < a.Norm;
	}
};

class AnnulusKMA : public HamerlyKMA {

protected:
	/**
	 * Index of the second closest center to point [index].
	 */
	LargeVector<int> SecondaryCenter;

	/**
	 * The norm of each data point.
	 */
	LargeVector<OPTFLOAT> PointNorm;

	/**
	 * Centroid with its norm.
	 */
	DynamicArray<CentroidNorm> CentroidNorms;

	/**
	 * Sort centers by increasing norm.
	 */
	void SortCenters();

	/**
	 * The outer AnnulusKMA loop shared between different versions.
	 */
	virtual bool OuterLoop(Array<OPTFLOAT> &vec, long *distanceCount);

	/**
	 * Init auxiliary data structures used by the algorithm
	 * given initial centroid coordinates.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	virtual void InitDataStructures( Array<OPTFLOAT> &vec);

	/**
	 * The Point-All-Ctrs function from Making K-means even faster.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 *  row_index - the index of the row in the Dataset
	 *  row - the data row as a float array
	 */
	virtual void PointAllCtrs(int lower, int upper, const Array<OPTFLOAT> &vec, const int row_index, const ThreadPrivateVector<OPTFLOAT > &row, long *count);
	virtual void PointAllCtrs(const Array<OPTFLOAT> &vec, const int row_index, const ThreadPrivateVector<OPTFLOAT > &row);

	/**
	 * Fill the CenterNorm array.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	void FillCentroidNorms(const Array<OPTFLOAT> &vec);

	/**
	 * Perform a binary search to find the centroids in radius.
	 */
	void BinarySearch(OPTFLOAT r, OPTFLOAT pointNorm, int &lower, int &upper);

	/**
	 * Check if all centroids in the range given by [lower, upper] are in radius.
	 * 	r - the radius
	 * 	pointNorm - the norm of a point
	 * 	lower - the lower bound of the range
	 * 	upper - the upper bound of the range
	 */
	void DbgVerifyRadius(OPTFLOAT r, OPTFLOAT pointNorm, int lower, int upper);

public:
	AnnulusKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR);
	AnnulusKMA(CentroidVector &aCV, StdDataset &Data, CentroidRepair *pR,HamerlySmallestDistances *pD);

	virtual double ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount);

	bool CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount);

};

#endif /* ANNULUSKMA_H_ */
