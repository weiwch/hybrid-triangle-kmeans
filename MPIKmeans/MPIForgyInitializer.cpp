/*
 * MPIForgyInitializer.cpp
 *
 *  Created on: Oct 9, 2015
 *      Author: wkwedlo
 */

#include <mpi.h>
#include "MPIForgyInitializer.h"
#include "../Util/Rand.h"

MPIForgyInitializer::MPIForgyInitializer(StdDataset &D,const char *fn,CentroidVector &aCV,int cl) : KMeansInitializer(D,aCV,cl),fname(fn) {

}


//#define TRACE_OBJNUMS
//#define TRACE_CENTROIDS


void MPIForgyInitializer::Init(Array<OPTFLOAT> &v) {
	int nCols=Data.GetColCount();
	int nTotalRows=Data.GetTotalRowCount();

	int Rank,Size;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);

	DynamicArray<float> FlatVector(nCols*nclusters);
	DynamicArray<int> ObjNums(nclusters);
	if (Rank==0) {
		for(int i=0;i<nclusters;i++)
			ObjNums[i]=(int)(Rand()*nTotalRows);
	}
	MPI_Bcast(ObjNums.GetData(),nclusters,MPI_INT,0,MPI_COMM_WORLD);
#ifdef TRACE_OBJNUMS
	for(int i=0;i<nclusters;i++)
		TRACE2("Process %d Obj %d\n",Rank,ObjNums[i]);
#endif

	StdDataset::LoadSelectedRows(fname,FlatVector,ObjNums);
	for(int i=0;i<nclusters*ncols;i++)
		v[i]=(OPTFLOAT)FlatVector[i];
#ifdef TRACE_CENTROIDS
	if (Rank==0)
		for(int i=0;i<nclusters*ncols;i++)
			TRACE3("Process %d, v[%d]=%f\n",Rank,i,v[i]);
#endif
}
