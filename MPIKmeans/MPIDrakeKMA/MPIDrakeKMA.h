#ifndef MPIDRAKEKMA_H_
#define MPIDRAKEKMA_H_

#include "../../Clust/DrakeKMA/DrakeKMA.h"
#include "../MPIForgyInitializer.h"
#include "../MPIKMAReducer.h"
#include "../DistributedNumaDataset.h"

class MPIDrakeKMA: public DrakeKMA {

protected:

	MPIKMAReducer *pReducer;

public:
	MPIDrakeKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR, int b, bool adaptiveDrake);

	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit, int &MaxB);
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont, int &MaxB);
	virtual ~MPIDrakeKMA();
};

#endif /* MPIDRAKEKMA_H_ */
