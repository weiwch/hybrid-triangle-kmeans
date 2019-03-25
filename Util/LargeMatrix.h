#ifndef __LARGEMATRIX_H
#define __LARGEMATRIX_H



#include "Matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern bool numa;

template <typename T> class LargeMatrix : public MatrixBase<T> {

protected:
	using MatrixBase<T>::nRows;
	using MatrixBase<T>::nCols;
	using MatrixBase<T>::nStride;
	using MatrixBase<T>::pBuff;

private:
	LargeMatrix(const LargeMatrix<T> *M);

public:
	void SetSize(int nR,int nC,int Alignment=16);
	LargeMatrix();
    LargeMatrix(int nR,int nC,int Alignment=16);
    ~LargeMatrix();
};


template <typename T> void LargeMatrix<T>::SetSize(int nR,int nC,int Alignment) {

	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();

	if (pBuff!=NULL) {
		pAlloc->Free(pBuff,(long)nRows*(long)nStride*sizeof(T));
	}
	nRows=nR;
	nCols=nC;

	nStride=nCols*sizeof(T);
	int div=nStride%Alignment;
	if (div)
		nStride+=(Alignment-div);
	nStride/=sizeof(T);
	pBuff=(T *)pAlloc->Alloc((long)nRows*(long)nStride*sizeof(T));
#pragma omp parallel for default(none)
		for(int i=0;i<nRows;i++) {
			for (int j=0;j<nCols;j++)
				(*this)(i,j)=(T)0;
		}
}

template <typename T> LargeMatrix<T>::LargeMatrix(int nR,int nC,int Alignment) {
	pBuff=NULL;
	SetSize(nR,nC,Alignment);
}

template <typename T> LargeMatrix<T>::LargeMatrix() {
	pBuff=NULL;
	nStride=nRows=nCols=0;
}


template <typename T> LargeMatrix<T>::~LargeMatrix() {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pAlloc->Free(pBuff,(long)nRows*(long)nStride*sizeof(T));
}

#endif
