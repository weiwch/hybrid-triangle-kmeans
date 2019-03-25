#ifndef MPIANNULUSKMA_H_
#define MPIANNULUSKMA_H_

#include "../../Clust/AnnulusKMA/AnnulusKMA.h"
#include "../MPIForgyInitializer.h"
#include "../MPIKMAReducer.h"
#include "../DistributedNumaDataset.h"

class MPIAnnulusKMA : public AnnulusKMA {

protected:

	MPIKMAReducer *pReducer;

public:
	MPIAnnulusKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR,HamerlySmallestDistances *pD);
	virtual ~MPIAnnulusKMA();

	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit);
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont);

};

#endif /* MPIANNULUSKMA_H_ */
