#ifndef PLUSPLUSINITIALIZER_H_
#define PLUSPLUSINITIALIZER_H_

#include "../KMeansInitializer.h"
#include "../../Util/LargeVector.h"

/**
 * The Kmeans++ initializer implementation.
 */
class PlusPlusInitializer: public KMeansInitializer {
public:
	void Init(Array<OPTFLOAT> &vec);
	void SetWeights(DynamicArray<OPTFLOAT> aWeights);
	PlusPlusInitializer(StdDataset &D, CentroidVector &aCV, int ncl);

protected:
	LargeVector<OPTFLOAT> ClosestDistances;
	LargeVector<OPTFLOAT> Weights;
	LargeVector<EXPFLOAT> AggregatedSum;
};

#endif /* PLUSPLUSINITIALIZER_H_ */
