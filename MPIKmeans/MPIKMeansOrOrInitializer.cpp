/*
 * MPIKMeansOrOrInitializer.cpp
 *
 *  Created on: May 15, 2017
 *      Author: wkwedlo
 */

#include "MPIKMeansOrOrInitializer.h"

MPIKMeansOrOrInitializer::MPIKMeansOrOrInitializer(DistributedNumaDataset &D,CentroidVector &aCV,int cl) : KMeansOrOrInitializer(D,aCV,cl,1.0) {
}

MPIKMeansOrOrInitializer::~MPIKMeansOrOrInitializer() {
}

