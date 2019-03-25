/*
 * YinyanClusterer.h
 *
 *  Created on: Nov 8, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_YINYANGCLUSTERER_H_
#define CLUST_YINYANGCLUSTERER_H_

#include "../Util/Array.h"
#include "../Util/StdDataset.h"

class YinyangClusterer {
public:
	YinyangClusterer() {}
	virtual OPTFLOAT FindAssignment(StdDataset &Data,DynamicArray<OPTFLOAT> &vec,DynamicArray<int> &Assignment,int nclusters)=0;
	virtual OPTFLOAT Recluster(StdDataset &Data,DynamicArray<int> &Assignment,int nclusters)=0;
	virtual const char *GetName()=0;
	virtual ~YinyangClusterer() {}
};

class KMeansYinyangClusterer : public YinyangClusterer {

public:
	virtual const char* GetName() {return "KMeans";}
	KMeansYinyangClusterer() {}
	virtual OPTFLOAT FindAssignment(StdDataset &Data,DynamicArray<OPTFLOAT> &vec,DynamicArray<int> &Assignment,int nclusters);
	virtual OPTFLOAT Recluster(StdDataset &Data,DynamicArray<int> &Assignment,int nclusters);
	virtual ~KMeansYinyangClusterer() {}

};

class SameSizeYinyangClusterer : public YinyangClusterer {

public:
	virtual const char* GetName() {return "SameSize";}
	SameSizeYinyangClusterer() {}
	virtual OPTFLOAT FindAssignment(StdDataset &Data,DynamicArray<OPTFLOAT> &vec,DynamicArray<int> &Assignment,int nclusters);
	virtual OPTFLOAT Recluster(StdDataset &Data,DynamicArray<int> &Assignment,int nclusters);
	virtual ~SameSizeYinyangClusterer() {}

};

#endif /* CLUST_YINYANGCLUSTERER_H_ */
