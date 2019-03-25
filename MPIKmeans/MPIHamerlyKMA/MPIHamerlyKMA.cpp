#include "MPIHamerlyKMA.h"
#include "MPIHamerlySmallestDistances.h"

void MPIHamerlyKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit) {
	pReducer->ReduceData(Centers, Counts, Fit);
}

void MPIHamerlyKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont) {
	pReducer->ReduceData(Centers, Counts, bCont);
}

MPIHamerlyKMA::MPIHamerlyKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR,
		HamerlySmallestDistances *pD)
		: HamerlyKMA(aCV, D, pRepair,pD) {
	pReducer = pR;
}

MPIHamerlyKMA::~MPIHamerlyKMA() {
}

