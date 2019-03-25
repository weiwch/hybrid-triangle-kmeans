/*
 * NumaAlloc.h
 *
 *  Created on: Mar 18, 2015
 *      Author: wkwedlo
 */

#ifndef UTIL_NUMAALLOC_H_
#define UTIL_NUMAALLOC_H_


#include <numaif.h>
#include <numa.h>
#include <sched.h>

class NUMAAllocator {

	static NUMAAllocator *Instance;
	NUMAAllocator() {HugePageThr=1.0;}

	double HugePageThr;
	void HugePagesAllowHeuristic(void *ptr,size_t Size);

public:
	void* Alloc(size_t Size);
	void Free(void *ptr,size_t Size);
	/// Thr==0 dont call madvise to disable transparent huge pages
	void SetHugePageThreshold(double Thr) {HugePageThr=Thr;}
	static NUMAAllocator *GetInstance();

};


extern inline int NodeOfAddr(void *ptr) {
	int numa_node = -1;
	get_mempolicy(&numa_node, NULL, 0, (void*)ptr, MPOL_F_NODE | MPOL_F_ADDR);
	return numa_node;

}


template <typename Array> double NumaLocalityofPrivateArray (Array & arr) {
	int Size=arr.GetSize();
	int DesiredCount=0;
	int CPU=sched_getcpu();
	int DesiredNode=numa_node_of_cpu(CPU);

	for(int i=0;i<Size;i++) {
		void *ptr=&(arr[i]);
		int RealNode=NodeOfAddr(ptr);
		if (DesiredNode==RealNode)
			DesiredCount++;
	}
	return (double)DesiredCount/(double)Size;
}

template <typename Matrix> double NumaLocalityofPrivateMatrix (Matrix & M) {

	int nRows=M.GetRowCount();
	int DesiredCount=0;
	int CPU=sched_getcpu();
	int DesiredNode=numa_node_of_cpu(CPU);

	for(int i=0;i<nRows;i++) {
		void *ptr=M(i);
		int RealNode=NodeOfAddr(ptr);
		if (DesiredNode==RealNode)
			DesiredCount++;
	}

	return (double)DesiredCount/(double)nRows;
}



template <typename Array> double NumaLocalityofArray (Array & arr) {

	int Size=arr.GetSize();
	int DesiredCount=0;

#pragma omp parallel default(none) firstprivate(Size) reduction(+:DesiredCount) shared(arr)
	{
		int CPU=sched_getcpu();
		int DesiredNode=numa_node_of_cpu(CPU);
#pragma omp for
		for(int i=0;i<Size;i++) {
			void *ptr=&(arr[i]);
			int RealNode=NodeOfAddr(ptr);
			if (DesiredNode==RealNode)
				DesiredCount++;
		}
	}
	return (double)DesiredCount/(double)Size;
}

template <typename Matrix> double NumaLocalityofMatrix (Matrix & M) {

	int nRows=M.GetRowCount();
	int DesiredCount=0;

#pragma omp parallel default(none) firstprivate(nRows) reduction(+:DesiredCount) shared(M)
	{
		int CPU=sched_getcpu();
		int DesiredNode=numa_node_of_cpu(CPU);
#pragma omp for
		for(int i=0;i<nRows;i++) {
			void *ptr=M(i);
			int RealNode=NodeOfAddr(ptr);
			if (DesiredNode==RealNode)
				DesiredCount++;
		}
	}
	return (double)DesiredCount/(double)nRows;
}

#endif /* UTIL_NUMAALLOC_H_ */
