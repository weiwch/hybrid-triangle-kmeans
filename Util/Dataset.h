#ifndef DATASET_H
#define DATASET_H

#include "Debug.h"
#include "Partition.h"
#include "Array.h"
#include "LargeMatrix.h"

/// Row of data - e.g. the learning vector
typedef Array<float> DataRow;

/// Dataset for learning algorithms, data stored as a matrix

/** This class represents dataset (learning set used in learning algorithms.
    The dataset is represented as two dimensional matrix. Each row corresponds
    to one object and each column to one variable. Elements are stored as 
    single precision real numbers
*/
class Dataset {

protected:
	int nRows,nCols,nStride;
	int nTotalRows;
	float *pMem;
	int Length;
	float *pBuff;
	DynamicArray<DataRow *> data;
	
	void WarmUp();
public:
	// Index operator e.g. D(i,j) returns element from row i and column j
	virtual float &operator()(int row,int col) {return pBuff[(long)row*nStride+(long)col];}
	// Index operator for read-only access
	virtual const float &operator()(int row,int col) const {return pBuff[(long)row*nStride+(long)col];}
	/// Number of rows i.e. the learning vectors
	virtual int GetRowCount() const {return nRows;}

	virtual int GetTotalRowCount() const {return nTotalRows;}
	
	/// Number of columns i.e. the variables
	virtual int GetColCount() const {return nCols;}
	
	/// Returns i-th row
	DataRow &GetRow(int i)  {return *data[i];}
	const float *GetRowNew(int i) const {
		ASSERT(i>=0);
		ASSERT(i<nRows);
		return pBuff+(long)i*nStride;
	}
	/// Returns i-th row for read-only access
	const DataRow &GetRow(int i) const {return *data[i];}
	Dataset();
	virtual ~Dataset();
	/// Loads data from binary file
	virtual void Load(char *fname);
	/// Loads parts of data from binary file (used in parallelized algorithm)
	virtual void Load(char *fname,int proc,int nprocs);
	
	virtual void Save(char* fname);
	
	void Filter(Dataset &D,Partition &P,int ClassNum);
	///Just reserves memory:
	int AllocateMemory(int rows, int cols);
	
	///Prints info about dataset:
	virtual void PrintInfo();
	
	void BenchmarkOld();
	void BenchmarkNew();

	virtual void Unload();
	
};
#endif

