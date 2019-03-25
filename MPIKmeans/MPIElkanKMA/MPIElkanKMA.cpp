#include "MPIElkanKMA.h"
#include "MPIElkanSmallestDistances.h"

void MPIElkanKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit) {
	pReducer->ReduceData(Centers, Counts, Fit);
}

void MPIElkanKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont) {
	pReducer->ReduceData(Centers, Counts, bCont);
}

MPIElkanKMA::MPIElkanKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR,ElkanSmallestDistances *pD)
		: ElkanKMA(aCV, D, pRepair,pD) {
	pReducer = pR;
}

MPIElkanKMA::~MPIElkanKMA() {
}

