/*
 * OpenMPKMAReducer.h
 *
 *  Created on: Feb 10, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_OPENMPKMAREDUCER_H_
#define CLUST_OPENMPKMAREDUCER_H_

#include "CentroidVector.h"
#include "../Util/LargeVector.h"
#include "../Util/Array.h"


/// Counts and Centers for all threads. Reducer is responsible for storing them
struct ReducerThreadData {
	ThreadPrivateVector<OPTFLOAT> Center;
	ThreadPrivateVector<int> Counts;
} __attribute__ ((aligned (64)));


class OpenMPKMAReducer {

protected:
	int NThreads;
	int nclusters;
	int ncols;
	int stride;

	/// Counts and Centers for all threads. Reducer is responsible for storing them
	DynamicArray<ReducerThreadData> ThreadData;

public:
	/// Returns the Counts array of a thread Thread
	ThreadPrivateVector<int> &GetThreadCounts(int Thread) {return ThreadData[Thread].Counts;}

	/// Returns the Center array of a thread Thread
	ThreadPrivateVector <OPTFLOAT> &GetThreadCenter(int Thread) {return ThreadData[Thread].Center;}

	/** Clears (zero) both Center and Counts array of a thread Thread
	 * For efficiency reasons should be called only from OpenMP thread Thread
	 */
	void ClearThreadData(int Thread);

	/// Adds Centers and Counts array of thread 0 tu arrays supplied as parameters
	void AddZeroToCenter(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts);

	/// Reduces all threads' Centers and Counts array to arrays of Thread 0
	virtual void ReduceToZero()=0;

	/**  Reduces all threads' Centers and Counts array to arrays supplied as parameters
		 Warning !!! Center and Counts must be cleared elsewhere !!! (by ClearArrays)
	*/
	virtual void ReduceToArrays(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts)=0;

	/// Clears (zero) both Center and Counts array supplied as parameters
	void ClearArrays(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts);

	OpenMPKMAReducer(CentroidVector &CV);
	virtual ~OpenMPKMAReducer() {}
};

class NaiveOpenMPReducer : public OpenMPKMAReducer {

public:
	virtual void ReduceToZero();
	virtual void ReduceToArrays(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts);
	NaiveOpenMPReducer(CentroidVector &CV);
};

class Log2OpenMPReducer : public OpenMPKMAReducer {
protected:
	void AddThreadArrays(int DstThr,int SrcThr);
public:
	virtual void ReduceToZero();
	virtual void ReduceToArrays(ThreadPrivateVector <OPTFLOAT> &Center,ThreadPrivateVector<int> &Counts);
	Log2OpenMPReducer(CentroidVector &CV);
};


#endif /* CLUST_OPENMPKMAREDUCER_H_ */
