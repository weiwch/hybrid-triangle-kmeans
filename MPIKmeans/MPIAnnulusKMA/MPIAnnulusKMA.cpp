#include "MPIAnnulusKMA.h"
#include "../MPIHamerlyKMA/MPIHamerlySmallestDistances.h"

void MPIAnnulusKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit) {
	pReducer->ReduceData(Centers, Counts, Fit);
}

void MPIAnnulusKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont) {
	pReducer->ReduceData(Centers, Counts, bCont);
}

MPIAnnulusKMA::MPIAnnulusKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR,HamerlySmallestDistances *pD)
		: AnnulusKMA(aCV, D, pRepair,pD) {
	pReducer = pR;

}

MPIAnnulusKMA::~MPIAnnulusKMA() {
}
