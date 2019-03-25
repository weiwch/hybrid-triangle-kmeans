/*
 * NumaDataset.cpp
 *
 *  Created on: Mar 28, 2013
 *      Author: wkwedlo
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <numa.h>
#include <omp.h>
#include <sched.h>
#include <limits.h>
#include "NumaDataset.h"
#include "NumaAlloc.h"
#include "FileException.h"

NumaDataset::NumaDataset()  {
	TotalSum=0.0;
}

NumaDataset::~NumaDataset() {
	pBuff=NULL;
	for(int i=0;i<data.GetSize();i++)
		delete data[i];
}




EXPFLOAT NumaDataset::WarmUp() {

	//printf("Node of pMappedMem: %d\n",NodeOfAddr(pMappedMem));
	int PrevNode=-1;

	/*for(int i=0;i<nRows;i++) {
		DataRow &row=GetRow(i);
		int Node=NodeOfAddr(row.GetData());
		if (PrevNode!=Node)
			printf("Row %d node %d\n",i,Node);
		PrevNode=Node;

	}*/
	EXPFLOAT Sum=0.0;
#pragma omp parallel reduction(+:Sum)
	{
#pragma omp for
		for(int i=0;i<nRows;i++)
		{
			const float *row=GetRowNew(i);
			for (int j=0;j<nCols;j++)
				Sum+=row[j];
		}
	}
	return Sum;
}

#ifdef _OPENMP
void NumaDataset::FindMinMaxRow(DynamicArray<int> &Max,DynamicArray<int> &Min)
{
	DynamicArray<int> CPUs,Nodes;
	int NThreads=omp_get_max_threads();
	Max.SetSize(NThreads);
	Min.SetSize(NThreads);
	CPUs.SetSize(NThreads);
	Nodes.SetSize(NThreads);
	for (int i=0;i<NThreads;i++) {
		Max[i]=-1;
		Min[i]=INT_MAX;
	}
#pragma omp parallel default(none) shared(Max,Min,CPUs,Nodes)
	{
		int id=omp_get_thread_num();
		CPUs[id]=sched_getcpu();
		Nodes[id]=numa_node_of_cpu(CPUs[id]);
		int maxid=-1,minid=INT_MAX;
#pragma omp for
		for(int i=0;i<nRows;i++) {
			if (maxid<i) maxid=i;
			if (minid>i) minid=i;
		}
		Max[id]=maxid;
		Min[id]=minid;
	}
//	for (int i=0;i<NThreads;i++)
//		printf("Thread %d CPU %d Node %d min %d max %d\n",i,CPUs[i],Nodes[i],Min[i],Max[i]);
}
#else
void NumaDataset::FindMinMaxRow(DynamicArray<int> &Max,DynamicArray<int> &Min)
{
	Max.SetSize(1);
	Min.SetSize(1);
	Min[0]=0;
	Max[0]=nRows-1;
}
#endif


void NumaDataset::ReadChunks(int fd,float *pData,int ChunkCount,int FirstRow,DynamicArray<struct iovec> &IOVecs) {
	ASSERT(ChunkCount<=IOVecs.GetSize());

	for(int i=0;i<ChunkCount;i++) {
		IOVecs[i].iov_base=pData;
		IOVecs[i].iov_len=nCols*sizeof(float);;
		data[FirstRow+i]=new DataRow(pData,nCols);
		pData+=nStride;

	}
	long BytesRead=readv(fd,IOVecs.GetData(),ChunkCount);
	long ExpectedRead=(long)ChunkCount*(long)nCols*sizeof(float);
	if(BytesRead!=ExpectedRead)
		throw FileException("Invalid number of bytes read by readv syscall");
}


void NumaDataset::LoadHeader(int fd) {
	struct stat st;
	if(fstat(fd,&st)==-1) {
		throw FileException("System call stat failed");
	}
	Length=st.st_size;

	if( read(fd,&nTotalRows,sizeof(nTotalRows))!=sizeof(nTotalRows)
	   || read(fd,&nCols,sizeof(nCols))!=sizeof(nCols))
		  throw FileException("read failed");

}

void NumaDataset::PartialLoad(char *fname,int StartRow,int nR) {
	nRows=nR;

	DynamicArray<int> MaxI,MinI;
	FindMinMaxRow(MaxI,MinI);

	nStride=nCols*sizeof(float);
	const int Alignment=16;
	int div=nStride%Alignment;
	if (div)
		nStride+=(Alignment-div);
	nStride/=sizeof(float);

	TRACE1("Stride in floats: %d\n",nStride);

	Data.SetSize(nRows,nStride,16);


	pBuff=Data.GetData();
	data.SetSize(nRows);




#ifdef _OPENMP
#pragma omp parallel default(none) shared(MaxI,MinI,stderr) firstprivate(StartRow,fname)
	{
			int id=omp_get_thread_num();
			int maxi=MaxI[id];
			int mini=MinI[id];
			int fdnew=open(fname,O_RDONLY);

			DynamicArray<struct iovec> IOVecs(1024);
			VERIFY(lseek64(fdnew,sizeof(int)*2+sizeof(float)*((long)StartRow+(long)mini)*(long)nCols,SEEK_SET)>0L);
			float *ptr=pBuff+(long)mini*(long)nStride;
			int tRows=maxi-mini+1;

			TRACE4("Thread %d fd: %d tRows %d mini %d\n",id,fdnew,tRows,mini);

			while(tRows>0) {
				int nChunks= tRows>IOVecs.GetSize() ? IOVecs.GetSize() : tRows;
				ReadChunks(fdnew,ptr,nChunks,mini,IOVecs);
				ptr+=((long)nChunks*(long)nStride);
				tRows-=nChunks;
				mini+=nChunks;
			}
			close(fdnew);
	}
#else
	int fdnew=open(fname,O_RDONLY);
	float *ptr=pBuff;
	VERIFY(lseek64(fdnew,sizeof(int)*2+sizeof(float)*(long)StartRow*(long)nCols,SEEK_SET)>0L);
	DynamicArray<struct iovec> IOVecs(1024);
	int tRows=nRows;
	int mini=0;
	while(tRows>0) {
		int nChunks= tRows>IOVecs.GetSize() ? IOVecs.GetSize() : tRows;
		ReadChunks(fdnew,ptr,nChunks,mini,IOVecs);
		ptr+=((long)nChunks*(long)nStride);
		tRows-=nChunks;
		mini+=nChunks;
	}

	close(fdnew);
#endif
	TotalSum=WarmUp();
	TRACE1("PartialLoad TotalSum=%Lf\n",TotalSum);
}



void NumaDataset::Load(char *fname) {
	int fd=open(fname,O_RDONLY);
	if (fd<0)
		throw FileException("Cannot open dataset");

	LoadHeader(fd);
	nRows=nTotalRows;
	close(fd);
	PartialLoad(fname,0,nTotalRows);
	//dbgVerifyLoad(fname);
}

void NumaDataset::GlobalFetchRow(int Position,DataRow &row) {
	ASSERT(Position>=0 && Position<GetRowCount());
	row=GetRow(Position);
}

void NumaDataset::dbgVerifyLoad(char *fname) {
#ifdef _DEBUG
	TRACE0("Loading StdDataset for verification\n");
	StdDataset D;
	D.Load(fname);
	TRACE4("Member data: nRows: %d nTotalRows: %d nCols: %d nStride: %d\n",nRows,nTotalRows,nCols,nStride);
	TRACE0("Veryfing dataset: old access method (const)\n");
	for(int i=0;i<nRows;i++) {
		const DataRow &row1=GetRow(i);
		const DataRow &row2=D.GetRow(i);
		for(int j=0;j<nCols;j++)
			if (row1[j]!=row2[j])
				TRACE2("Difference at row %d col %d\n",i,j);
	}

	TRACE0("Veryfing dataset: old access method (no const)\n");
	for(int i=0;i<nRows;i++) {
		DataRow &row1=GetRow(i);
		DataRow &row2=D.GetRow(i);
		for(int j=0;j<nCols;j++)
			if (row1[j]!=row2[j])
				TRACE2("Difference at row %d col %d\n",i,j);
	}

	TRACE0("Veryfing dataset: new access method\n");
	for(int i=0;i<nRows;i++) {
		const float *row1=GetRowNew(i);
		const DataRow &row2=D.GetRow(i);
		for(int j=0;j<nCols;j++)
			if (row1[j]!=row2[j])
				TRACE2("Difference at row %d col %d\n",i,j);
	}

	TRACE0("Veryfing dataset: new versus old method\n");
	for(int i=0;i<nRows;i++) {
		const float *row1=GetRowNew(i);
		const DataRow &row2=GetRow(i);
		for(int j=0;j<nCols;j++)
			if (row1[j]!=row2[j])
				TRACE2("Difference at row %d col %d\n",i,j);
	}

#endif
}
