#include <limits>
#include "CentroidVector.h"
#include "../Util/FileException.h"




double CentroidVector::ComputeMSE(Array<OPTFLOAT> &vec, const Dataset &Data) {
	ASSERT(vec.GetSize()==ncols*nclusters);
	double fit=0.0;
	for(int i=0;i<Data.GetRowCount();i++) {
		const DataRow &row=Data.GetRow(i);
		OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
		for(int j=0;j<nclusters;j++) {
			OPTFLOAT ssq=SquaredDistance(j,vec,row);
			if (ssq<minssq)
				minssq=ssq;
		}
		fit+=minssq;
	}
	return fit/(double)Data.GetRowCount();
}

double CentroidVector::ComputeMSE(Array<OPTFLOAT> &vec, const Dataset &Data, const DynamicArray<int> &Assignment) {
	ASSERT(vec.GetSize()==ncols*nclusters);
	double fit=0.0;
#pragma omp parallel for default(none) reduction(+:fit) shared(vec,Assignment,Data)
	for(int i=0;i<Data.GetRowCount();i++) {
		const DataRow &row=Data.GetRow(i);
		OPTFLOAT minssq=SquaredDistance(Assignment[i],vec,row);
		fit+=minssq;
	}
	return fit/(double)Data.GetRowCount();
}


void CentroidVector::ComputeClusterSSE(const Array<OPTFLOAT> &vec, const Dataset &Data,Array<OPTFLOAT> &ClustSSE) {
	int nRows=Data.GetRowCount();
	for(int i=0;i<nclusters;i++)
		ClustSSE[i]=0.0;
	for(int i=0;i<nRows;i++) {
		const DataRow &row=Data.GetRow(i);
		OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
		int minj=-1;
		for(int j=0;j<nclusters;j++) {
			OPTFLOAT ssq=SquaredDistance(j,vec,row);
			if (ssq<minssq) {
				minssq=ssq;
				minj=j;
			}
		}
		ClustSSE[minj]+=minssq;
	}
}

void CentroidVector::ClassifyDataset(Array<OPTFLOAT> &v,const Dataset &Data, Array<int> &ClNums) {
	int nRows=Data.GetRowCount();
	int nCols=Data.GetColCount();

#pragma omp parallel for default(none) shared(Data,v,ClNums) firstprivate(nRows)
	for(int i=0;i<nRows;i++) {
		const DataRow &row=Data.GetRow(i);
		OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
		int bestj=-1;
		for(int j=0;j<nclusters;j++) {
			OPTFLOAT ssq=SquaredDistance(j,v,row);
			if (ssq<minssq) {
				minssq=ssq;
				bestj=j;
			}
		}
		ClNums[i]=bestj;
	}
}

void CentroidVector::ClassifyDataset(Array<OPTFLOAT> &v,const Dataset &Data, Array<int> &ClNums,Array<OPTFLOAT> &Distances) {
	int nRows=Data.GetRowCount();
	int nCols=Data.GetColCount();

#pragma omp parallel for default(none) shared(Data,v,Distances,ClNums) firstprivate(nRows)
	for(int i=0;i<nRows;i++) {
		const DataRow &row=Data.GetRow(i);
		OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
		int bestj=-1;
		for(int j=0;j<nclusters;j++) {
			OPTFLOAT ssq=SquaredDistance(j,v,row);
			if (ssq<minssq) {
				minssq=ssq;
				bestj=j;
			}
		}
		ClNums[i]=bestj;
		Distances[i]=minssq;
	}

}




void CentroidVector::FindObjectsInCluster(const Array<OPTFLOAT> &vec,const Dataset &Data,int Num,
			DynamicArray<int> &ObjNums) {
	int nRows=Data.GetRowCount();
	int nCols=Data.GetColCount();
	int Cntr=0;
	ObjNums.SetSize(nRows);

	for(int i=0;i<Data.GetRowCount();i++) {
		const DataRow &row=Data.GetRow(i);
		OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
		int bestj=-1;
		for(int j=0;j<nclusters;j++) {
			OPTFLOAT ssq=SquaredDistance(j,vec,row);
			if (ssq<minssq) {
				minssq=ssq;
				bestj=j;
			}
		}
		if (bestj==Num) {
			ObjNums[Cntr++]=i;
		}
	}
	ObjNums.SetSize(Cntr);
}


static int arrsize;
#pragma omp threadprivate(arrsize)

static int SortFunction(const void *p1,const void *p2)
{
	const OPTFLOAT *pp1=(OPTFLOAT *)p1;
	const OPTFLOAT *pp2=(OPTFLOAT *)p2;
	for (int i=0;i<arrsize;i++)
	{	if (pp1[i] < pp2[i]) return 1;
		if (pp1[i] > pp2[i]) return -1;
	}
	return 0;
}

void CentroidVector::Sort(Array<OPTFLOAT> &vec)
{
	ASSERT(vec.GetSize()==nclusters*ncols);
	arrsize=ncols;
	qsort(vec.GetData(),nclusters,ncols*sizeof(OPTFLOAT),SortFunction);
}

void CentroidVector::ComputeCenters(Array<OPTFLOAT> &vec, const Dataset &Data, const Array<int>Nums)
{
	DynamicArray<int> Cntr(nclusters);
	int nRows=Data.GetRowCount();

	for(int i=0;i<nclusters;i++)
		Cntr[i]=0;

	for(int i=0;i<vec.GetSize();i++)
		vec[i]=0.0;

	for(int i=0;i<nRows;i++) {
		int ClustNum=Nums[i];
		Cntr[ClustNum]++;
		const DataRow &row=Data.GetRow(i);
		int Start=ClustNum*ncols;
		for(int j=0;j<ncols;j++)
			vec[Start+j]+=row[j];

	}
	for(int i=0;i<nclusters;i++)
		for(int j=0;j<ncols;j++)
			vec[i*ncols+j]/=(double)Cntr[i];

}
