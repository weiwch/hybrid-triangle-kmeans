#include "../Clust/KMAlgorithm.h"
#include <mpi.h>

extern inline MPI_Datatype OptFloatType() {
	if (sizeof(OPTFLOAT)==4)
		return MPI_FLOAT;

	return MPI_DOUBLE;
}

extern inline MPI_Datatype ExpFloatType() {
	if (sizeof(EXPFLOAT)==8)
		return MPI_DOUBLE;
	return MPI_LONG_DOUBLE;
}
