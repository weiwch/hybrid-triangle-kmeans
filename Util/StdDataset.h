#ifndef STANDARDIZEDDATASET_H_
#define STANDARDIZEDDATASET_H_

#include "Dataset.h"

class StdDataset : public Dataset
{


protected:

	double TraceCov;

	DynamicArray<float> Max;
	DynamicArray<float> Min;


	// Std and Mean measured from the dataset
	DynamicArray<float> OrigMeans;
	DynamicArray<float> OrigStdDevs;
	
	// Std and Mean used in conversion (originally 1 and 0)
	DynamicArray<float> Means;
	DynamicArray<float> StdDevs;
	/// Chris: Implement this !!! (fix for MPI)
	virtual void ComputeGlobalSums() {;};



public:

	static void LoadSelectedRows(const char *fname, DynamicArray<float> &FlatVector, const DynamicArray<int> &ObjNums);


	void LoadAndProcess(char *fname, int rank = 0, int size = 1);
	
	double GetCovMatrixTrace() const {return TraceCov;}

	/// Standarizes column in a dataset 
	void Standarize(int Column);
	
	/// Returns the mean of the original (not standarized) column;	
	float GetMean(int Column) const {return OrigMeans[Column];}
	
	float GetMin(int Column) const {return Min[Column];}

	float GetMax(int Column) const {return Max[Column];}

	/// Returns the standard deviation of the (not standarized) column;
	float GetStdDev(int Column) const {return OrigStdDevs[Column];}
	
	/// Converts raw value to standarized value for given column
	float StandarizeValue(int Column,float Value) const;
	/// Converts standarized value back to raw value for given column
	float DeStandarizeValue(int Column,float Value) const;
	
	/// Saves means and stdevs to file;
	int Serialize(FILE *f);
	/// Loads means adn stdevs from file;
	int DeSerialize(FILE *f);	
	

	/// Construct a dataset from vector of centroids, needed by YinyangKMA.
	StdDataset(Array<OPTFLOAT> &vec,int ncols,int nclusters);
	StdDataset(DynamicArray< DynamicArray<float> > &Centers);
	StdDataset();
	virtual ~StdDataset();
};

#endif /*STANDARDIZEDDATASET_H_*/
