#ifndef KMEANSINITIALIZER_
#define KMEANSINITIALIZER_
#include "../Util/StdDataset.h"
#include "CentroidVector.h"

#ifndef EXPFLOAT
#define EXPFLOAT double
#endif

class KMeansInitializer {
	
protected:
	int ncols;
	StdDataset &Data;
	CentroidVector &CV;
	int nclusters;
	
public:
	virtual void Init(Array<OPTFLOAT> &v)=0;
	KMeansInitializer(StdDataset &D,CentroidVector &aCV,int cl);
	void RepairCentroid(Array<OPTFLOAT> &vec,int clnum);
	virtual ~KMeansInitializer() {}
};

class ForgyInitializer : public KMeansInitializer {
public:
	void Init(Array<OPTFLOAT> &v);
	ForgyInitializer(StdDataset &D,CentroidVector &aCV,int cl);
};

class MinDistInitializer : public KMeansInitializer {

	int Trials;
public:
	void Init(Array<OPTFLOAT> &v);
	MinDistInitializer(StdDataset &D,CentroidVector &aCV,int cl,int Tr);
};

class RandomInitializer : public KMeansInitializer {

	DynamicArray<int> PartCounts;
	
public:
	void Init(Array<OPTFLOAT> &v);
	RandomInitializer(StdDataset &D,CentroidVector &aCV,int cl);
};


class FileInitializer : public KMeansInitializer {
protected:
	DynamicArray<OPTFLOAT> vec;
public:
	void Init(Array<OPTFLOAT> &v);
	FileInitializer(StdDataset &D,CentroidVector &aCV,int cl,const char *fname);
};
#endif /*KMEANSINITIALIZER_*/
