/*
 * MPIKMAReducer.cpp
 *
 *  Created on: Oct 12, 2015
 *      Author: wkwedlo
 */

#include <string.h>
#include <sys/times.h>
#include <time.h>

#include "MPIUtils.h"
#include "MPIKMAReducer.h"
#include "../Clust/OpenMPKMAReducer.h"
#include "../Util/StdOut.h"



MPIKMAReducer *MPIKMAReducer::CreateReducer(const char *name,int nclus,int ncols,int param) {


	if (name==NULL || !strcmp(name,"simple"))
		return new SimpleReducer(nclus,ncols,param);

	if (!strcmp(name,"nonblocking"))
		return new NonBlockingReducer(nclus,ncols);

	if (!strcmp(name,"packed"))
		return new PackedReducer(nclus,ncols);


	return NULL;
}


void MPIKMAReducer::Benchmark(int Reps) {
#ifdef _OPENMP
	kma_printf("Benchmark of %s MPI reducer + Log2OpenMP reducer (%d repetitions)\n",GetName(),Reps);
#else
	kma_printf("Benchmark of %s MPI reducer (%d repetitions)\n",GetName(),Reps);
#endif

	CentroidVector CV(nClusters,nCols);
	ThreadPrivateVector<OPTFLOAT> Centers;
	Centers.SetSize(nClusters*nCols);
	ThreadPrivateVector<int> Counts;
	Counts.SetSize(nClusters);
	bool bCont=true;

	for(int i=0;i<Centers.GetSize();i++)
		Centers[i]=(OPTFLOAT)0.0;
	for(int i=0;i<nClusters;i++)
		Counts[i]=0;
	MPI_Barrier(MPI_COMM_WORLD);
	timespec tpstart_monotonic,tpend_monotonic;

#ifdef _OPENMP
	OpenMPKMAReducer *pOMPReducer=new Log2OpenMPReducer(CV);
#endif
	clock_gettime(CLOCK_MONOTONIC, &tpstart_monotonic);
	for (int i=0;i<Reps;i++) {
#ifdef _OPENMP
		pOMPReducer->ReduceToArrays(Centers,Counts);
#endif
		ReduceData(Centers,Counts,bCont);
	}
	clock_gettime(CLOCK_MONOTONIC, &tpend_monotonic);
	double extime2_monotonic=(double)tpend_monotonic.tv_sec-(double)tpstart_monotonic.tv_sec+1e-9*((double)tpend_monotonic.tv_nsec-(double)tpstart_monotonic.tv_nsec);
	kma_printf("Benchmark execution time: %g seconds\n",extime2_monotonic);
#ifdef _OPENMP
	delete pOMPReducer;
#endif
}


SimpleReducer::SimpleReducer(int nclus,int ncols,int param) :MPIKMAReducer(nclus,ncols) {
	Portion=param;
	TRACE1("SimpleReducer::Portion=%d\n",Portion);
}


void SimpleReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit) {

	if (Portion==-1) {
		MPI_Allreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}	else {
		for(int i=0;i<nClusters;i+=Portion)
			MPI_Allreduce(MPI_IN_PLACE,Centers.GetData()+i*nCols,Portion*nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}
		MPI_Allreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE,&Fit,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD);

}


void SimpleReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit, EXPFLOAT &Fit_pure) {

	if (Portion==-1) {
		MPI_Allreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}	else {
		for(int i=0;i<nClusters;i+=Portion)
			MPI_Allreduce(MPI_IN_PLACE,Centers.GetData()+i*nCols,Portion*nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}
		MPI_Allreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE,&Fit,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE,&Fit_pure,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD);

}

void SimpleReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont) {
	int ConvertedCont=bCont;

	if (Portion==-1) {
		MPI_Allreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}	else {
		for(int i=0;i<nClusters;i+=Portion)
			MPI_Allreduce(MPI_IN_PLACE,Centers.GetData()+i*nCols,Portion*nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}
	MPI_Allreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&ConvertedCont,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
	bCont=ConvertedCont;
}

void SimpleReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit,int &MaxB) {
	if (Portion==-1) {
		MPI_Allreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}	else {
		for(int i=0;i<nClusters;i+=Portion)
			MPI_Allreduce(MPI_IN_PLACE,Centers.GetData()+i*nCols,Portion*nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}
	MPI_Allreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&Fit,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&MaxB,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
}


void SimpleReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont,int &MaxB) {
	int ConvertedCont=bCont;
	if (Portion==-1) {
		MPI_Allreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}	else {
		for(int i=0;i<nClusters;i+=Portion)
			MPI_Allreduce(MPI_IN_PLACE,Centers.GetData()+i*nCols,Portion*nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	}
	MPI_Allreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&ConvertedCont,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
	bCont=ConvertedCont;
	MPI_Allreduce(MPI_IN_PLACE,&MaxB,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
}



NonBlockingReducer::NonBlockingReducer(int nclus,int ncols) :MPIKMAReducer(nclus,ncols) {
}

void NonBlockingReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit) {


	MPI_Request reqtab[3];
	MPI_Status stattab[3];

	MPI_Iallreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[0]);
	MPI_Iallreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD,&reqtab[1]);
	MPI_Iallreduce(MPI_IN_PLACE,&Fit,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[2]);

	MPI_Waitall(3,reqtab,stattab);

}

void NonBlockingReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit, EXPFLOAT &Fit_pure) {


	MPI_Request reqtab[4];
	MPI_Status stattab[4];

	MPI_Iallreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[0]);
	MPI_Iallreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD,&reqtab[1]);
	MPI_Iallreduce(MPI_IN_PLACE,&Fit,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[2]);
	MPI_Iallreduce(MPI_IN_PLACE,&Fit_pure,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[3]);

	MPI_Waitall(4,reqtab,stattab);

}

void NonBlockingReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit,int &MaxB) {
	MPI_Request reqtab[4];
	MPI_Status stattab[4];

	MPI_Iallreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[0]);
	MPI_Iallreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD,&reqtab[1]);
	MPI_Iallreduce(MPI_IN_PLACE,&Fit,1,ExpFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[2]);
	MPI_Iallreduce(MPI_IN_PLACE,&MaxB,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD,&reqtab[3]);
	MPI_Waitall(4,reqtab,stattab);

}



void NonBlockingReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont) {
	int ConvertedCont=bCont;


	MPI_Request reqtab[3];
	MPI_Status stattab[3];

	MPI_Iallreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[0]);
	MPI_Iallreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD,&reqtab[1]);
	MPI_Iallreduce(MPI_IN_PLACE,&ConvertedCont,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD,&reqtab[2]);
	MPI_Waitall(3,reqtab,stattab);

	bCont=ConvertedCont;
}

void NonBlockingReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont,int &MaxB) {
	int ConvertedCont=bCont;


	MPI_Request reqtab[4];
	MPI_Status stattab[4];

	MPI_Iallreduce(MPI_IN_PLACE,Centers.GetData(),Centers.GetSize(),OptFloatType(),MPI_SUM,MPI_COMM_WORLD,&reqtab[0]);
	MPI_Iallreduce(MPI_IN_PLACE,Counts.GetData(),Counts.GetSize(),MPI_INT,MPI_SUM,MPI_COMM_WORLD,&reqtab[1]);
	MPI_Iallreduce(MPI_IN_PLACE,&ConvertedCont,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD,&reqtab[2]);
	MPI_Iallreduce(MPI_IN_PLACE,&MaxB,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD,&reqtab[3]);
	MPI_Waitall(4,reqtab,stattab);

	bCont=ConvertedCont;
}


static int GlobalNClust,GlobalNCols;

void reduction_function(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
	//ASSERT(*datatype==MPI_BYTE);
	char *src=(char *)invec;
	char *dst=(char *)inoutvec;

	const OPTFLOAT * __restrict__ ofsrc=(OPTFLOAT *)src;
	OPTFLOAT * __restrict__ ofdst=(OPTFLOAT *)dst;
	int nElems=GlobalNClust*GlobalNCols;
	#pragma omp simd
	for(int i=0;i<nElems;i++)
		ofdst[i]+=ofsrc[i];
	int nBytes=nElems*sizeof(OPTFLOAT);
	src+=nBytes;
	dst+=nBytes;

	const int * __restrict__ isrc=(int *)src;
	int * __restrict__ idst=(int *)dst;
	#pragma omp simd
	for(int i=0;i<GlobalNClust;i++)
		idst[i]+=isrc[i];
	nBytes=GlobalNClust*sizeof(int);
	src+=nBytes;
	dst+=nBytes;

	EXPFLOAT *efsrc=(EXPFLOAT *)src;
	EXPFLOAT *efdst=(EXPFLOAT *)dst;
	*efdst+=*efsrc;

}

void reduction_function_with_cont(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
	//ASSERT(*datatype==MPI_BYTE);
	char *src=(char *)invec;
	char *dst=(char *)inoutvec;

	const OPTFLOAT * __restrict__ ofsrc=(OPTFLOAT *)src;
	OPTFLOAT * __restrict__ ofdst=(OPTFLOAT *)dst;
	int nElems=GlobalNClust*GlobalNCols;
	#pragma omp simd
	for(int i=0;i<nElems;i++)
		ofdst[i]+=ofsrc[i];
	int nBytes=nElems*sizeof(OPTFLOAT);
	src+=nBytes;
	dst+=nBytes;

	const int * __restrict__ isrc=(int *)src;
	int * __restrict__ idst=(int *)dst;
	#pragma omp simd
	for(int i=0;i<GlobalNClust;i++)
		idst[i]+=isrc[i];
	nBytes=GlobalNClust*sizeof(int);
	src+=nBytes;
	dst+=nBytes;

	int *bsrc=(int *)src;
	int *bdst=(int *)dst;
	*bdst |=*bsrc;

}



PackedReducer::PackedReducer(int nclus,int ncols) : MPIKMAReducer(nclus,ncols){
	GlobalNClust=nclus;
	GlobalNCols=ncols;

	int nBytes=GetPackSize();
	Buffer.SetSize(nBytes);
	MPI_Op_create(reduction_function,1,&Op);
	MPI_Op_create(reduction_function_with_cont,1,&OpWithCont);
	MPI_Type_contiguous(nBytes,MPI_BYTE,&Type);
	MPI_Type_commit(&Type);
}

PackedReducer::~PackedReducer() {
	MPI_Op_free(&Op);
	MPI_Op_free(&OpWithCont);
	MPI_Type_free(&Type);
}

void PackedReducer::UnpackData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit) {
	char *ptr=Buffer.GetData();
	int nBytes=Centers.GetSize()*sizeof(OPTFLOAT);
	memcpy(Centers.GetData(),ptr,nBytes);
	ptr+=nBytes;

	nBytes=Counts.GetSize()*sizeof(int);
	memcpy(Counts.GetData(),ptr,nBytes);
	ptr+=nBytes;

	nBytes=sizeof(EXPFLOAT);
	memcpy(&Fit,ptr,nBytes);

}

void PackedReducer::UnpackData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit, EXPFLOAT &Fit_pure) {
	char *ptr=Buffer.GetData();
	int nBytes=Centers.GetSize()*sizeof(OPTFLOAT);
	memcpy(Centers.GetData(),ptr,nBytes);
	ptr+=nBytes;

	nBytes=Counts.GetSize()*sizeof(int);
	memcpy(Counts.GetData(),ptr,nBytes);
	ptr+=nBytes;

	nBytes=sizeof(EXPFLOAT);
	memcpy(&Fit,ptr,nBytes);
	ptr+=nBytes;

	memcpy(&Fit_pure,ptr,nBytes);

}

void PackedReducer::UnpackData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &Cont) {
	char *ptr=Buffer.GetData();
	int nBytes=Centers.GetSize()*sizeof(OPTFLOAT);
	memcpy(Centers.GetData(),ptr,nBytes);
	ptr+=nBytes;

	nBytes=Counts.GetSize()*sizeof(int);
	memcpy(Counts.GetData(),ptr,nBytes);
	ptr+=nBytes;

	nBytes=sizeof(bool);
	memcpy(&Cont,ptr,nBytes);

}

void PackedReducer::PackData(const ThreadPrivateVector<OPTFLOAT> &Centers,const ThreadPrivateVector<int> &Counts,const EXPFLOAT &Fit) {
	char *ptr=Buffer.GetData();
	int nBytes=Centers.GetSize()*sizeof(OPTFLOAT);
	memcpy(ptr,Centers.GetData(),nBytes);
	ptr+=nBytes;

	nBytes=Counts.GetSize()*sizeof(int);
	memcpy(ptr,Counts.GetData(),nBytes);
	ptr+=nBytes;

	nBytes=sizeof(EXPFLOAT);
	memcpy(ptr,&Fit,nBytes);

}

void PackedReducer::PackData(const ThreadPrivateVector<OPTFLOAT> &Centers, const ThreadPrivateVector<int> &Counts, const EXPFLOAT &Fit, const EXPFLOAT &Fit_pure)
{
	char *ptr=Buffer.GetData();
	int nBytes=Centers.GetSize()*sizeof(OPTFLOAT);
	memcpy(ptr,Centers.GetData(),nBytes);
	ptr+=nBytes;

	nBytes=Counts.GetSize()*sizeof(int);
	memcpy(ptr,Counts.GetData(),nBytes);
	ptr+=nBytes;

	nBytes=sizeof(EXPFLOAT);
	memcpy(ptr,&Fit,nBytes);
	ptr+=nBytes;

	memcpy(ptr,&Fit_pure,nBytes);
}

void PackedReducer::PackData(const ThreadPrivateVector<OPTFLOAT> &Centers,const ThreadPrivateVector<int> &Counts,const bool &Cont) {
	char *ptr=Buffer.GetData();
	int nBytes=Centers.GetSize()*sizeof(OPTFLOAT);
	memcpy(ptr,Centers.GetData(),nBytes);
	ptr+=nBytes;

	nBytes=Counts.GetSize()*sizeof(int);
	memcpy(ptr,Counts.GetData(),nBytes);
	ptr+=nBytes;

	nBytes=sizeof(bool);
	memcpy(ptr,&Cont,nBytes);

}

void PackedReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,bool &bCont) {
	PackData(Centers,Counts,bCont);
	MPI_Allreduce(MPI_IN_PLACE,Buffer.GetData(),1,Type,OpWithCont,MPI_COMM_WORLD);
	UnpackData(Centers,Counts,bCont);
}

void PackedReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit) {
	PackData(Centers,Counts,Fit);
	MPI_Allreduce(MPI_IN_PLACE,Buffer.GetData(),1,Type,Op,MPI_COMM_WORLD);
	UnpackData(Centers,Counts,Fit);
}


void PackedReducer::ReduceData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<int> &Counts,EXPFLOAT &Fit, EXPFLOAT &Fit_pure) {
	PackData(Centers,Counts,Fit,Fit_pure);
	MPI_Allreduce(MPI_IN_PLACE,Buffer.GetData(),1,Type,Op,MPI_COMM_WORLD);
	UnpackData(Centers,Counts,Fit,Fit_pure);
}
