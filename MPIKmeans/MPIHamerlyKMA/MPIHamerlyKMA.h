#ifndef MPIHAMERLYKMA_H_
#define MPIHAMERLYKMA_H_

#include "../../Clust/HamerlyKMA/HamerlyKMA.h"
#include "../MPIForgyInitializer.h"
#include "../MPIKMAReducer.h"
#include "../DistributedNumaDataset.h"

class MPIHamerlyKMA : public HamerlyKMA {

protected:

	MPIKMAReducer *pReducer;

public:
	MPIHamerlyKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR,HamerlySmallestDistances *pD);

	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit);

	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont);

	virtual ~MPIHamerlyKMA();
};

#endif /* MPIHAMERLYKMA_H_ */
