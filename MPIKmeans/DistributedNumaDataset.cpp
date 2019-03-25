/*
 * DistributedNumaDataset.cpp
 *
 *  Created on: Oct 8, 2015
 *      Author: wkwedlo
 */
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "DistributedNumaDataset.h"
#include "../Util/FileException.h"

DistributedNumaDataset::DistributedNumaDataset() {
	int Size,Rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);

	Offsets.SetSize(Size);
	Counts.SetSize(Size);
}


void DistributedNumaDataset::ComputeDataPositions(int Size) {


	for(int Rank=0;Rank<Size;Rank++) {
		int SubSize=nTotalRows/Size;
		int SubRemainder=nTotalRows % Size;
		int Offset;
		if (Rank<SubRemainder) {
			SubSize++;
			Offset=SubSize*Rank;
		} else {
			Offset=SubSize*Rank+SubRemainder;
		}
		Counts[Rank]=SubSize;
		Offsets[Rank]=Offset;
	}
	//TRACE3("MPI Process %d row offset :%d row count %d\n",Rank,RowOffset,RowCount);
}

void DistributedNumaDataset::GlobalFetchRow(int Position,DataRow &row) {
	ASSERT(Position>=0 && Position<nTotalRows);
	int Rank,Size,Root=-1;
	MPI_Comm_size(MPI_COMM_WORLD,&Size);
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);

	for(int i=0;i<Size;i++) {
		int RowOffset,RowCount;
		if (Offsets[i]<=Position && Position<Offsets[i]+Counts[i])
			Root=i;
	}
	if (Rank==Root) {
		int MyRow=Position-Offsets[Rank];
		ASSERT(MyRow>=0);
		ASSERT(MyRow<=GetRowCount());
		row=GetRow(MyRow);
	}
	MPI_Bcast(row.GetData(),nCols,MPI_FLOAT,Root,MPI_COMM_WORLD);
}


void DistributedNumaDataset::Load(char *fname) {
	int Size,Rank;

	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);

	int fd=open(fname,O_RDONLY);
	if (fd<0)
		throw FileException("Cannot open dataset");

	LoadHeader(fd);
	close(fd);
	ComputeDataPositions(Size);
	PartialLoad(fname,Offsets[Rank],Counts[Rank]);
	float TrueTotalSum;
	MPI_Allreduce(&TotalSum,&TrueTotalSum,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	TotalSum=TrueTotalSum;
}
