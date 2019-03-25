#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include "Dataset.h"
#include "FileException.h"
#include "../Util/Debug.h"

using namespace std;

Dataset::Dataset()
{
	pBuff=NULL;
	pMem=NULL;
	nRows=0;
	nCols=0;
	
}


void Dataset::Unload() 
{
	if (pBuff!=NULL) {
		for(int i=0;i<nRows;i++) 
			delete data[i];
		delete [] pMem;
		pBuff=NULL;
	}
}

Dataset::~Dataset()
{
	Unload();
}


/** Loads dataset from a binary file. The
   The format is as follows: number of rows (32bit int),
   number of columns (32bit int), data for row1, row2, ....
   
   Each row should contain float (32bit) numbers
*/

void Dataset::Load(char *fname) 
{
	Load(fname,0,1);
}


volatile float Sum=0.0f;

/*
 * Access all memory-mapped dataset, to assure that
 * it is really loaded
 */
void Dataset::WarmUp() {
	for(int i=0;i<nRows;i++)
	{
		DataRow row=GetRow(i);
		for (int j=0;j<nCols;j++)
			Sum+=row[j];
	}
}

void Dataset::Filter(Dataset &D,Partition &P,int ClassNum) {
	int nRows=P.GetObjCount(ClassNum);
	int nCols=D.GetColCount();
	AllocateMemory(nRows,nCols);
	for(int i=0;i<nRows;i++) {
		int ObjNum=P.GetObjNum(ClassNum,i);
		const DataRow &src=D.GetRow(ObjNum);
		DataRow &dst=GetRow(i);
		for (int j=0;j<nCols;j++)
			dst[j]=src[j];
	}
}


void Dataset::Load(char *fname,int Rank,int nProcs)
{
	// Open file
	FILE *fd=fopen(fname,"rb");
	if (fd==NULL)
		throw FileException("Cannot open dataset");

	// Find and set length
	struct stat st;
	if(fstat(fileno(fd),&st)==-1) {
		throw FileException("System call stat failed");
	}
	Length=st.st_size;

	// Read the number of rows and columns of data (first two ints from file)
	if( fread(&nTotalRows,sizeof(nTotalRows),1,fd)!=1
	   || fread(&nCols,sizeof(nCols),1,fd)!=1)
		  throw FileException("fread failed");

	// Calculate the shift required to skip a given row (in bytes)
	nStride=nCols*sizeof(float);

	// ?
	int div=nStride%16;
	if (div)
		nStride+=(16-div);
	nStride/=4;
	
	// Calculate the number of rows each process should read, and the remainder
	int SubSize=nTotalRows/nProcs;
	int SubRemainder=nTotalRows % nProcs;

	// ?
	int Offset;
	if (Rank<SubRemainder) {
		SubSize++;
		Offset=SubSize*Rank*nCols;
	} else {
		Offset=(SubSize*Rank+SubRemainder)*nCols;
	}
	nRows=SubSize;
	fseek(fd,Offset*sizeof(float)+2*sizeof(int),SEEK_SET);
	pMem=new float [nRows*nStride+4];
	float *pData=pMem;
	while((unsigned long)pData & 0x0f)
		pData++;
	pBuff=pData;
	TRACE2("Dataset::Load Columns %d Stride: %d\n",nCols,nStride);
	TRACE3("Dataset::Load - Proc: %d Offset: %d Size: %d\n",Rank,Offset/nCols,nRows);

	// Read the data from the file
	data.SetSize(nRows);
	for(int i=0;i<nRows;i++) {
		if (fread(pData,sizeof(float),nCols,fd)!=(unsigned)nCols)
			throw FileException("Cannot read row");			
		data[i]=new DataRow(pData,nCols);
		pData+=nStride;
//		TRACE2("Row %d Addr %x\n",i,pData);
	}

	// Close the file and warm up caches
	fclose(fd);
	WarmUp();				
}

void Dataset::Save(char* fname) {
	FILE *fd=fopen(fname,"wb");
	if (fd==NULL)
		throw FileException("Cannot open dataset");

	if( fwrite(&nTotalRows,sizeof(nTotalRows),1,fd)!=1
	   || fwrite(&nCols,sizeof(nCols),1,fd)!=1)
		  throw FileException("fread failed");
	   	
	for(int i=0;i<nRows;i++) {
		DataRow &row = GetRow(i);
		if (fwrite(row.GetData(),sizeof(float),nCols,fd)!=(unsigned)nCols)
			throw FileException("Cannot write row");			
//		TRACE2("Row %d Addr %x\n",i,pData);
	}
	fclose(fd);	
}

int Dataset::AllocateMemory(int rows, int cols) {
	nTotalRows = nRows = rows;
	nCols = cols;
	
	nStride=nCols*sizeof(float);
	int div=nStride%16;
	if (div)
		nStride+=(16-div);
	nStride/=4;
	
	pMem=new float [nRows*nStride+4];
	float *pData=pMem;
	while((unsigned long)pData & 0x0f)
		pData++;	
	pBuff=pData;
	data.SetSize(nRows);
	for(int i=0;i<nRows;i++) {
		data[i]=new DataRow(pData,nCols);
		pData+=nStride;
//		TRACE2("Row %d Addr %x\n",i,pData);
	}
	return 0;
}

void Dataset::PrintInfo() {
	printf("Rows: %d\nColumns: %d\n", nRows, nCols);
	for(int i=0;i<GetRowCount();i++) {
		const DataRow &row=GetRow(i);
		for (int j=0;j<GetColCount();j++)
			printf("%5.3f ",row[j]);
		printf("\n");
	}
}

void Dataset::BenchmarkOld() {
	timespec tp1,tp2;
	clock_gettime(CLOCK_MONOTONIC,&tp1);

	double TotalSum=0.0;
#pragma omp parallel for  reduction(+:TotalSum)
	for(int i=0;i<nRows;i++) {
		double Sum=0.0;
		const DataRow &row=GetRow(i);
		for (int j=0;j<nCols;j++)
			Sum+=row[j];
		TotalSum+=Sum;
	}
	clock_gettime(CLOCK_MONOTONIC,&tp2);
	double extime=1000.0*((double)tp2.tv_sec+1e-9*(double)tp2.tv_nsec-(double)tp1.tv_sec-1e-9*(double)tp1.tv_nsec);
	printf("Benchmark Old: Total sum: %f\n",TotalSum);
	printf("Benchmark Old execution time: %f mseconds\n",extime);
}


void Dataset::BenchmarkNew() {

	timespec tp1,tp2;
	clock_gettime(CLOCK_MONOTONIC,&tp1);

	double TotalSum=0.0;
#pragma omp parallel for  reduction(+:TotalSum)
	for(int i=0;i<nRows;i++) {
		double Sum=0.0;
		const float *row=GetRowNew(i);
		for (int j=0;j<nCols;j++)
			Sum+=row[j];
		TotalSum+=Sum;
	}
	clock_gettime(CLOCK_MONOTONIC,&tp2);
	printf("Benchmark New: Total sum: %f\n",TotalSum);
	double extime=1000.0*((double)tp2.tv_sec+1e-9*(double)tp2.tv_nsec-(double)tp1.tv_sec-1e-9*(double)tp1.tv_nsec);
	printf("Benchmark New execution time: %f mseconds\n",extime);
}

