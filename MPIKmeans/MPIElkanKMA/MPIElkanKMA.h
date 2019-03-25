#ifndef MPIELKANKMA_H_
#define MPIELKANKMA_H_

#include "../../Clust/ElkanKMA/ElkanKMA.h"
#include "../MPIForgyInitializer.h"
#include "../MPIKMAReducer.h"
#include "../DistributedNumaDataset.h"

class MPIElkanKMA : public ElkanKMA {

protected:

	MPIKMAReducer *pReducer;

public:
	MPIElkanKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR,ElkanSmallestDistances *pD);

	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit);
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont);
	virtual ~MPIElkanKMA();
};

#endif /* MPIELKANKMA_H_ */
