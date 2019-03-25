/*
 * MPIKMAReducer.h
 *
 *  Created on: Oct 12, 2015
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPIKMAREDUCER_H_
#define MPIKMEANS_MPIKMAREDUCER_H_

#include "../Clust/KMAlgorithm.h"
#include <mpi.h>

class MPIKMAReducer {
protected:
	int nCols,nClusters;
public:

	/// TODO Reduction operation without Fit required for triangle based algs.
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit)=0;
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont)=0;
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit,int &MaxB)=0;
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont,int &MaxB)=0;
	virtual const char *GetName()=0;
	MPIKMAReducer(int nclus,int ncols) {nCols=ncols; nClusters=nclus;}
	virtual ~MPIKMAReducer() {}
	void Benchmark(int Reps);

	static MPIKMAReducer *CreateReducer(const char *name,int nclus,int ncols,int param);
};

class SimpleReducer : public MPIKMAReducer {
	int Portion;
public:
	SimpleReducer(int nclus,int ncols,int param);
	virtual const char *GetName() {return "simple blocking";}
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit);
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont);
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit,int &MaxB);
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont,int &MaxB);
};


class NonBlockingReducer : public MPIKMAReducer {


public:
	NonBlockingReducer(int nclus,int ncols);
	virtual const char *GetName() {return "simple nonblocking";}
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit);
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont);
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit,int &MaxB);
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont,int &MaxB);
};

class PackedReducer : public MPIKMAReducer {

protected:

	MPI_Op Op,OpWithCont;
	MPI_Datatype Type;
	ThreadPrivateVector<char> Buffer;

	/// We assume that sizeof(EXPFLOAT) >= sizeof(bool)
	int GetPackSize() const {return sizeof(OPTFLOAT)*nClusters*nCols+sizeof(int)*nClusters+sizeof(EXPFLOAT); }


	void PackData(const ThreadPrivateVector<OPTFLOAT> &Centers,const ThreadPrivateVector<int> &Counts,const EXPFLOAT &Fit);
	void PackData(const ThreadPrivateVector<OPTFLOAT> &Centers,const ThreadPrivateVector<int> &Counts,const bool &Cont);

	void UnpackData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit);
	void UnpackData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &Cont);

public:
	virtual ~PackedReducer();
	PackedReducer(int nclus,int ncols);
	virtual const char *GetName() {return "packed";}
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit);
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont);
	// reduction for DrakeKMA not implemented yet
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit,int &MaxB) {}
	virtual void ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont,int &MaxB) {}
};

#endif /* MPIKMEANS_MPIKMAREDUCER_H_ */
