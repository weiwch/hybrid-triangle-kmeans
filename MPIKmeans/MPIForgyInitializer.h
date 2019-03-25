/*
 * MPIForgyInitializer.h
 *
 *  Created on: Oct 9, 2015
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPIFORGYINITIALIZER_H_
#define MPIKMEANS_MPIFORGYINITIALIZER_H_

#include "../Clust/KMeansInitializer.h"

class MPIForgyInitializer: public KMeansInitializer {


	const char *fname;

public:
	void Init(Array<OPTFLOAT> &v);
	MPIForgyInitializer(StdDataset &D,const char *fname,CentroidVector &aCV,int cl);
	virtual ~MPIForgyInitializer() {}
};

#endif /* MPIKMEANS_MPIFORGYINITIALIZER_H_ */
