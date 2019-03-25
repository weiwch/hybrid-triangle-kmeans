/*
 * NumaDataset.h
 *
 *  Created on: Mar 28, 2013
 *      Author: wkwedlo
 */

#ifndef NUMADATASET_H_
#define NUMADATASET_H_
#include <sys/uio.h>

#include "StdDataset.h"
#include "../Util/LargeMatrix.h"

#ifndef EXPFLOAT
#define EXPFLOAT double
#endif

class NumaDataset : public StdDataset{

	LargeMatrix<float> Data;

protected:


	EXPFLOAT TotalSum;

	EXPFLOAT WarmUp();
	void FindMinMaxRow(DynamicArray<int> &Max,DynamicArray<int> &Min);
	void PartialLoad(char *fname,int StartRow,int nR);
	void LoadHeader(int fd);

	void ReadChunks(int fd,float *pData,int ChunkCount,int FirstRow,DynamicArray<struct iovec> &IOVecs);

	void dbgVerifyLoad(char *fname);
public:
	virtual void GlobalFetchRow(int Position,DataRow &row);
	double NumaLocality() {return NumaLocalityofMatrix(Data);}
	float GetTotalSum() {return TotalSum;}
	virtual void Load(char *fname);
	NumaDataset();
	virtual ~NumaDataset();
};

#endif /* NUMADATASET_H_ */
