#include "KMeansWithoutMSE.h"
#include "../../Util/Debug.h"
#include "../../Util/StdOut.h"
#include <time.h>

KMeansWithoutMSE::KMeansWithoutMSE(CentroidVector &CV, StdDataset &Data, int &IterCount)
	: cv(CV), data(Data), iterCount(IterCount) {
}

void KMeansWithoutMSE::RunKMeansWithoutMSE(Array<OPTFLOAT> &vec, const int verbosity) {
	ASSERT(vec.GetSize() == cv.GetNCols() * cv.GetNClusters());

	long distanceCount = 0;
	double distanceCalculationRatio = 0;
	double distanceCalculationRatioSum = 0;
	timespec tpstart_monotonic;
	clock_gettime(CLOCK_MONOTONIC, &tpstart_monotonic);

	InitDataStructures(vec);
	int i = 0;
	for(;;i++) {
		distanceCount = 0;
		timespec tpend_monotonic;

		bool cont = CorrectWithoutMSE(vec, &distanceCount);
		clock_gettime(CLOCK_MONOTONIC, &tpend_monotonic);
	  	double iterTime=(double)tpend_monotonic.tv_sec-(double)tpstart_monotonic.tv_sec+1e-9*((double)tpend_monotonic.tv_nsec-(double)tpstart_monotonic.tv_nsec);

		distanceCalculationRatio = (double) (distanceCount / ((double) (data.GetTotalRowCount() * cv.GetNClusters()))) * 100;
		distanceCalculationRatioSum += distanceCalculationRatio;

		if (verbosity > 3) {
			kma_printf("i: %d, dcr: %g, itime: %g seconds\n", i, distanceCalculationRatio, iterTime);
		}


		if (i > 0 && !cont) {
			break;
		}
	}

	iterCount += i + 1;

	if (verbosity > 0) {
		kma_printf("Avg distance calculation ratio: %g\n", distanceCalculationRatioSum / iterCount);
	}
}
