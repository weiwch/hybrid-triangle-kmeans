#include "StdDataset.h"
#include "../Util/FileException.h"

#include <cmath>

StdDataset::StdDataset()
{
}

StdDataset::~StdDataset()
{
}

StdDataset::StdDataset(Array<OPTFLOAT> &vec,int ncols,int nclusters) {
	TraceCov=-1.0;
	nCols=ncols;
	nRows=nTotalRows=nclusters;
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
			for(int j=0;j<nCols;j++)
				pData[j]=(float)vec[i*ncols+j];
			data[i]=new DataRow(pData,nCols);
			pData+=nStride;
	}
	//PrintInfo();
}

StdDataset::StdDataset(DynamicArray< DynamicArray<float> > &Centers) {
	TraceCov=-1.0;
	nCols=Centers[0].GetSize();
	nRows=nTotalRows=Centers.GetSize();
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
			DynamicArray<float> Center=Centers[i];
			for(int j=0;j<nCols;j++)
				pData[j]=(float)Center[j];
			data[i]=new DataRow(pData,nCols);
			pData+=nStride;
	}
	//PrintInfo();
}


void StdDataset::LoadAndProcess(char *fname, int rank, int size) {
	Dataset::Load(fname, rank, size);
	
	Max.SetSize(nCols);
	Min.SetSize(nCols);

	OrigMeans.SetSize(nCols);
	OrigStdDevs.SetSize(nCols);
	Means.SetSize(nCols);
	StdDevs.SetSize(nCols);
	for(int i=0;i<nCols;i++) {
		OrigMeans[i]=Means[i]=OrigStdDevs[i]=0.0f;
		StdDevs[i]=1.0f;
	}
	
	const DataRow &row0=GetRow(0);
	for(int i=0;i<nCols;i++)
		Max[i]=Min[i]=row0[i];

// Compute means (trivial) and std devs using the trick :V(X)=E(X^2)-E(X)^2 
	for(int i=0;i<nRows;i++) {
		const DataRow &rData=GetRow(i);
		for(int j=0;j<nCols;j++) {
			OrigMeans[j]+=rData[j];
			// First E(X^2)
			OrigStdDevs[j]+=(rData[j]*rData[j]);
			if (rData[j]>Max[j])
				Max[j]=rData[j];
			if (rData[j]<Min[j])
				Min[j]=rData[j];
		}
	}
	// Chris: Hook for parallel version
	ComputeGlobalSums();
	TraceCov=0.0;
	for(int i=0;i<nCols;i++) {
		OrigMeans[i]/=(float)nTotalRows;
		// Now we have E(X^2)
		OrigStdDevs[i]/=(float)nTotalRows;
		// Now we have biased variance
		OrigStdDevs[i]-=(OrigMeans[i]*OrigMeans[i]);
		// Now we have unbiased variance		
		OrigStdDevs[i]*=((float)nTotalRows/((float)nTotalRows-1.0f));
		TraceCov+=OrigStdDevs[i];
		// And finally standard deviation		
		OrigStdDevs[i]=std::sqrt(OrigStdDevs[i]);
		TRACE3("Column%d: mean: %5.2f std: %5.2f\n",i,OrigMeans[i],
				OrigStdDevs[i]);

	}
	printf("StdDataset: trace of a covariance matrix: %f\n",TraceCov);

}

void StdDataset::LoadSelectedRows(const char *fname, DynamicArray<float> &FlatVector, const DynamicArray<int> &ObjNums) {

	int nCols,nTotalRows;



	FILE *fd=fopen(fname,"rb");
	if (fd==NULL)
		throw FileException("Cannot open dataset");


	if( fread(&nTotalRows,sizeof(nTotalRows),1,fd)!=1
		  || fread(&nCols,sizeof(nCols),1,fd)!=1)
			 throw FileException("fread failed");

	int nRequestedRows=ObjNums.GetSize();

	FlatVector.SetSize(nCols*nRequestedRows);

	for(int i=0;i<nRequestedRows;i++) {
		int Offset=i*nCols;
		int RequestedRow=ObjNums[i];
		fseek(fd,sizeof(int)*2+sizeof(float)*(long)RequestedRow*(long)nCols,SEEK_SET);
		fread(FlatVector.GetData()+Offset,sizeof(float),nCols,fd);
	}
	fclose(fd);
}


void StdDataset::Standarize(int Col) 
{
	for(int i=0;i<GetRowCount();i++)
		(*this)(i,Col)=((*this)(i,Col)-OrigMeans[Col])/OrigStdDevs[Col];

	// Conversion will work from now
	TRACE2("Standarizer: OrigMeans[%d]=%f\n", Col, OrigMeans[Col]);
	TRACE2("Standarizer: OrigStdDevs[%d]=%f\n", Col, OrigStdDevs[Col]);
	Means[Col]=OrigMeans[Col];
	StdDevs[Col]=OrigStdDevs[Col];	
}

float StdDataset::StandarizeValue(int Column,float Value) const
{	
	return (Value-Means[Column])/StdDevs[Column];
}

float StdDataset::DeStandarizeValue(int Column,float Value) const
{
	return Value*StdDevs[Column]+Means[Column];
}

int StdDataset::Serialize(FILE *f)
{
	unsigned int size;


		size = OrigMeans.GetSize();
		if (fwrite(&size, sizeof(int), 1, f)<1)
		 throw FileException("Standarizer: Could not write size\n");
		if (fwrite(OrigMeans.GetData(), sizeof(float), size, f)<size)
		 throw FileException("Standarizer: Could not write OrigMeans\n");;
		if (fwrite(OrigStdDevs.GetData(), sizeof(float), size, f)<size)
		 throw FileException("Standarizer: Could not write OrigStdDevs\n");
		if (fwrite(Means.GetData(), sizeof(float), size, f)<size)
		 throw FileException("Standarizer: Could not write Means\n");
		if (fwrite(StdDevs.GetData(), sizeof(float), size, f)<size)
		 throw FileException("Standarizer: Could not write StdDevs\n");

	return 0;

}

int StdDataset::DeSerialize(FILE *f)
{	
	unsigned int size;

		
		if (fread(&size, sizeof(int), 1, f)<1)
		 throw FileException("Standarizer: Could not read item count\n");
		TRACE1("size: %d\n", size);
		OrigMeans.SetSize(size);
		OrigStdDevs.SetSize(size);
		Means.SetSize(size);
		StdDevs.SetSize(size);
		if (fread(OrigMeans.GetData(), sizeof(int), size, f)<size)
		 throw FileException("Standarizer: Could not read OrigMeans\n");	
		if (fread(OrigStdDevs.GetData(), sizeof(int), size, f)<size)
		 throw FileException("Standarizer: Could not read OrigStdDevs\n");
		if (fread(Means.GetData(), sizeof(int), size, f)<size)
		 throw FileException("Standarizer: Could not read Means\n");	
		if (fread(StdDevs.GetData(), sizeof(int), size, f)<size)
		 throw FileException("Standarizer: Could not read StdDevs\n");	

	return 0;
}

