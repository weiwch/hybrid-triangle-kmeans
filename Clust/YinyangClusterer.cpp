/*
 * YinyanClusterer.cpp
 *
 *  Created on: Nov 8, 2016
 *      Author: wkwedlo
 */

#include "KMAlgorithm.h"
#include "SameSizeKMA.h"
#include "CentroidVector.h"
#include "YinyangClusterer.h"



OPTFLOAT KMeansYinyangClusterer::FindAssignment(StdDataset &Data,DynamicArray<OPTFLOAT> &vec,DynamicArray<int> &Assignment,int nclusters) {
	CentroidDeterministicRepair Repair(Data,nclusters);
	CentroidVector CV(nclusters,Data.GetColCount());
	NaiveKMA Alg(CV,Data,&Repair);
	OPTFLOAT MSE=Alg.RunKMeans(vec,0,0.0);
	CV.ClassifyDataset(vec,Data,Assignment);
	return MSE;
}
OPTFLOAT KMeansYinyangClusterer::Recluster(StdDataset &Data,DynamicArray<int> &Assignment,int nclusters) {
	CentroidDeterministicRepair Repair(Data,nclusters);
	CentroidVector CV(nclusters,Data.GetColCount());
	NaiveKMA Alg(CV,Data,&Repair);
	DynamicArray<OPTFLOAT> vec(CV.GetNClusters()*CV.GetNCols());
	CV.ComputeCenters(vec,Data,Assignment);
	OPTFLOAT MSE=Alg.RunKMeans(vec,0,0.0);
	CV.ClassifyDataset(vec,Data,Assignment);
	return MSE;
}

OPTFLOAT SameSizeYinyangClusterer::FindAssignment(StdDataset &Data,DynamicArray<OPTFLOAT> &vec,DynamicArray<int> &Assignment,int nclusters) {
	CentroidVector CV(nclusters,Data.GetColCount());
	SameSizeKMA KMA(CV,Data);
	OPTFLOAT MSE=KMA.TrainIterative(vec,Assignment,10,0);
	return MSE;
}

OPTFLOAT SameSizeYinyangClusterer::Recluster(StdDataset &Data,DynamicArray<int> &Assignment,int nclusters) {
	CentroidVector CV(nclusters,Data.GetColCount());
	SameSizeKMA KMA(CV,Data);
	DynamicArray<OPTFLOAT> vec(CV.GetNClusters()*CV.GetNCols());
	CV.ComputeCenters(vec,Data,Assignment);
	OPTFLOAT MSE=KMA.TrainIterativeFromAssignment(vec,Assignment,10,0);
	return MSE;
}
