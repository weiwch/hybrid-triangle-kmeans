#ifndef OPENMP_H_
#define OPENMP_H_

#ifdef _OPENMP
#include <omp.h>
#else
extern inline int omp_get_num_threads() {
	return 1;
}

extern inline int omp_get_thread_num() {
	return 0;
}

extern inline int omp_get_max_threads() {
	return 1;
}
#endif

#endif /* OPENMP_H_ */
