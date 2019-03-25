#include "MPIDrakeKMA.h"

void MPIDrakeKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, EXPFLOAT &Fit, int &MaxB) {
	pReducer->ReduceData(Centers, Counts, Fit, MaxB);
}

void MPIDrakeKMA::ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers, ThreadPrivateVector<int> &Counts, bool &bCont, int &MaxB) {
	pReducer->ReduceData(Centers, Counts, bCont, MaxB);
}

MPIDrakeKMA::MPIDrakeKMA(CentroidVector &aCV, DistributedNumaDataset &D, CentroidRepair *pRepair, MPIKMAReducer *pR, int b, bool adaptiveDrake) :
		DrakeKMA(aCV, D, pRepair, b, adaptiveDrake) {
	pReducer = pR;
}

MPIDrakeKMA::~MPIDrakeKMA() {
}

