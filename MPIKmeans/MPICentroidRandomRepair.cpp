/*
 * MPICentroidRandomRepair.cpp
 *
 *  Created on: Feb 3, 2016
 *      Author: wkwedlo
 */

#include <mpi.h>
#include "../Util/Rand.h"
#include "MPICentroidRandomRepair.h"
#include "MPIRank0StdOut.h"

#define TRACE_REPAIR



MPICentroidRandomRepair::MPICentroidRandomRepair(DistributedNumaDataset &D,int ncl) : CentroidRepair(ncl), Data(D) {

}


void MPICentroidRandomRepair::RepairVec(Array<OPTFLOAT> &vec,int Pos) {
	int Rank,Size,Source;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);
	if (Rank==0)
		Source=Rand()*Data.GetTotalRowCount();
	MPI_Bcast(&Source,1,MPI_INT,0,MPI_COMM_WORLD);
#ifdef TRACE_REPAIR
	TRACE2("MPICentroidRandomRepair Process: %d Source is: %d\n",Rank,Source);
#endif
	int nCols=Data.GetColCount();
	DynamicArray<float> Buffer(nCols);
	Data.GlobalFetchRow(Source,Buffer);
	for(int i=0;i<nCols;i++)
		vec[Pos*nCols+i]=Buffer[i];
	kma_printf("Empty centroid %d repaired by row %d\n",Pos,Source);
}

