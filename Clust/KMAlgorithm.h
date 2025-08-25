#ifndef KMALGORITHM_H_
#define KMALGORITHM_H_

#include "CentroidVector.h"
#include "CentroidRepair.h"
#include "KMeansInitializer.h"
#include "../Util/StdDataset.h"
#include "../Util/LargeVector.h"
#include "KMeansWithoutMSE/KMeansWithoutMSE.h"
#include "OpenMPKMAReducer.h"
#include "KMeansReportWriter.h"

#ifndef EXPFLOAT
#define EXPFLOAT double
#endif


#ifndef OMPDYNAMIC
#define OMPDYNAMIC
#endif

/// The abstract class representing the K-means algorithm

/**
 * The solution vector (vec parameter of RunKMeans) stores coordinates of the cluster centroids
 * in the following manner
 * (coordinates of the first centroid,coordinates of the second centroid, ....,coordinates of the last centroid).
 */

class KMeansAlgorithm
{
protected:
	int ncols;     		/// Number of columns in dataset
	int nclusters; 		/// Number of clusters K
	int IterCount;      /// K-means iteration counter
	CentroidVector &CV;
	StdDataset &Data;  /// Dataset used to train K-Means
	CentroidRepair *pRepair;
	KMeansReportWriter *pReportWriter;


	static int LastIter;
	static double LastTime;


	/// Random perturbation of the solution vector used by k-means with stochastic relaxation
	void PeturbVector(Array<OPTFLOAT> &vec, double rtime,double MaxTime);

	/// Init auxiliary data structures used by the algorithm given initial centroid coordinates
	virtual void InitDataStructures(Array<OPTFLOAT> &vec) {}

	virtual void PrintIterInfo(int i, double BestFit, double Rel,double Avoided, double iterTime);
	virtual void PrintAvgAvoidance(double Avoided);

public:
	static int GetLastIter() {return LastIter;}
	static double GetLastTime() {return LastTime;}

	KMeansAlgorithm(CentroidVector &aCV,StdDataset &Data,CentroidRepair *pR);
	/// Runs the KMeans algorithm (with default MinRel until termination)
	virtual double RunKMeans(Array<OPTFLOAT> &vec,int verbosity,double MinRel=-1,int MaxIter=0);

	/// Runs the KMeans algorithm with stochastic relaxation
	virtual double RunKMeansWithSR(Array<OPTFLOAT> &vec, bool print,double MaxTime);

	/// single iteration of the K-Means
	virtual double ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount=NULL)=0;

	virtual void PrintNumaLocalityInfo() {}
	int GetIterCount() const {return IterCount;}
	int GetNCols() const {return ncols;}
	int GetNClusters() const {return nclusters;}
	void SetReportWriter(KMeansReportWriter *pR) {pReportWriter=pR;}
	void ResetIterCount() {IterCount=0;}
	virtual ~KMeansAlgorithm() {}
};



/// Local data (centers and counts) for each OpenMP thread
struct NaiveThreadData {
	ThreadPrivateVector<OPTFLOAT> row;
}__attribute__ ((aligned (64)));

/// Naive straightforward version of the K-Means algorithm


/**
 * This is the naive (based on definition) version of the K-Means algorithm.
 * However, this version is very well parallelized for NUMA architectures,
 * using OpenMP. It scales very well (tested on the 64-core mordor3 server)
 */
class NaiveKMA : public KMeansAlgorithm, public KMeansWithoutMSE {
public:
	LargeVector<int> Assignment;
	double Lambda;
	double prev_fit = 0.0; // previous fit value
	double rand_rate = 0.0; // rate of randomization
protected:

	/// The centroids of the new clusters

	/**
	 * New cluster centroids computed by ComputeMSEAndCorrect method stored as
	 * coordinates of the first centroid, second centroid, ..., k-th centroid
	 */
	ThreadPrivateVector<OPTFLOAT> Center;

	 /// Number of objects allocated to each center (needed to update centroid coordinates).
	ThreadPrivateVector<int> Counts;

	DynamicArray<NaiveThreadData> OMPData;

	OpenMPKMAReducer *pOMPReducer;

	/// Future Data reduction for version MPI, now empty
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit, EXPFLOAT &Fit_pure) {}
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont) {}
	double ComputeMSEAndCorrectImpl(Array<OPTFLOAT> &vec, long *distanceCount=NULL);
	double ComputeMSEAndCorrectRandImpl(Array<OPTFLOAT> &vec, long *distanceCount=NULL);
public:
	virtual void PrintNumaLocalityInfo();
	virtual bool CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount);
	virtual void InitDataStructures(Array<OPTFLOAT> &vec);

	virtual double ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount=NULL);
	NaiveKMA(CentroidVector &aCV,StdDataset &Data,CentroidRepair *pR, double lambda=0.0);
	~NaiveKMA();
};

#endif /*KMALGORITHM_H_*/
