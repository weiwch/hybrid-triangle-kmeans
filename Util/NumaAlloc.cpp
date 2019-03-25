#include <stdlib.h>
#include <numa.h>
#include <sys/mman.h>
#include "NumaAlloc.h"
#include "OpenMP.h"
#include "Debug.h"
#include "StdOut.h"
#include <algorithm>

NUMAAllocator *NUMAAllocator::Instance;

NUMAAllocator *NUMAAllocator::GetInstance() {
	if (Instance==NULL)
		Instance=new NUMAAllocator;
	return Instance;
}

#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif

const int HugePageSize=2*1024*1024;

//#define TRACE_NUMA_ALLOC

void NUMAAllocator::HugePagesAllowHeuristic(void *ptr,size_t Size) {
	int maxnodes=std::min(omp_get_max_threads(),numa_num_configured_nodes());
	double HugePagesPerNode=(double)Size/(double)maxnodes/(double)HugePageSize;
	bool bAllow=false;
	if (maxnodes==1 || HugePagesPerNode>HugePageThr)
		bAllow=true;
#ifdef TRACE_NUMA_ALLOC
	kma_printf("Memory allocation of %f MB, predicted huge pages: %f, forcing no hugepages: %d ",(double)Size/1024.0/1024.0,HugePagesPerNode,!bAllow);
#endif
	if (!bAllow) {
		int Ret=madvise(ptr,Size, MADV_NOHUGEPAGE);
#ifdef TRACE_NUMA_ALLOC
		kma_printf("madvise result: %d\n",Ret);
#endif
	} else  {
#ifdef TRACE_NUMA_ALLOC
		kma_printf("\n");
#endif
	}
}


void* NUMAAllocator::Alloc(size_t Size) {
	void *ptr=mmap(NULL,Size,PROT_READ | PROT_WRITE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
	HugePagesAllowHeuristic(ptr,Size);
	return ptr;
}

void NUMAAllocator::Free(void *ptr,size_t Size) {
	munmap(ptr,Size);
}


