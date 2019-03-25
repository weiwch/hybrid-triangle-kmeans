#ifndef UTIL_MATRIX_H_
#define UTIL_MATRIX_H_

#include "Debug.h"
#include "NumaAlloc.h"
#include <stdio.h>


template <typename T> class MatrixBase {

protected:
	int nRows,nCols,nStride;
	T * __restrict__ pBuff;
public:
	T *operator()(int i);
	T &operator()(int i,int j);
    const T & operator()(int i,int j) const;

    int GetRowCount() const {return nRows;}
    int GetColCount() const {return nCols;}
    int GetStride() const {return nStride;}

	T *GetData();
	MatrixBase();
	virtual ~MatrixBase() {;}
	void fprintf(FILE *pF);
};

template <typename T> MatrixBase<T>::MatrixBase() {
	pBuff=NULL;
	nRows=0;
	nCols=0;
	nStride=0;
}
template <typename T> T *MatrixBase<T>::GetData() {
	ASSERT(pBuff!=NULL);
	return pBuff;
}


template <typename T> T *MatrixBase<T>::operator()(int i) {
	ASSERT(pBuff!=NULL);
	ASSERT(i>=0);
	ASSERT(i<nRows);
	return pBuff+(long)i*nStride;
}


template <typename T> T& MatrixBase<T>::operator()(int i,int j)
{
	ASSERT(pBuff!=NULL);
	ASSERT(i>=0);
	ASSERT(i<nRows);
	ASSERT(j>=0);
	ASSERT(j<nCols);
	return pBuff[(long)i*nStride+(long)j];
}

template <typename T> const T& MatrixBase<T>::operator()(int i,int j) const
{
	ASSERT(i>=0);
	ASSERT(pBuff!=NULL);
	ASSERT(i<nRows);
	ASSERT(j>=0);
	ASSERT(j<nCols);
	return pBuff[(long)i*nStride+(long)j];
}


template <typename T> void MatrixBase<T>::fprintf(FILE *pF) {
	for(int i=0;i<nRows;i++) {
		for(int j=0;j<nCols;j++)
			::fprintf(pF,"%g ",(double)((*this)(i,j)));
		::fprintf(pF,"\n");
	}
}

template <typename T> class Matrix : public MatrixBase<T> {

protected:
	using MatrixBase<T>::nRows;
	using MatrixBase<T>::nCols;
	using MatrixBase<T>::nStride;
	using MatrixBase<T>::pBuff;
public:
    Matrix<T> &operator=(const Matrix<T> &M);

	Matrix(int nR,int nC,int Alignment=1);
    Matrix(int nR,int nC,T *pData,int Alignment=1);
    Matrix(const Matrix<T> &M);
    Matrix();
    void SetSize(int nR,int nC,int Alignment=1);
    void SetZero();
    ~Matrix();
};

template <typename T> void Matrix<T>::SetZero() {
	for(int i=0;i<nRows;i++)
		for(int j=0;j<nCols;j++)
			(*this)(i,j)=(T)0;
}

template <typename T> void Matrix<T>::SetSize(int nR,int nC,int Alignment) {

	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();

	if (pBuff!=NULL)
		pAlloc->Free(pBuff,sizeof(T)*(long)nRows*(long)nStride);


	nRows=nR;
	nCols=nC;

	nStride=nCols*sizeof(T);
	int div=nStride%Alignment;
	if (div)
		nStride+=(Alignment-div);
	nStride/=sizeof(T);


	pBuff=(T *)pAlloc->Alloc((long)nRows*(long)nStride*sizeof(T));
	SetZero();
}


template <typename T> Matrix<T>::Matrix(const Matrix<T> &M) {
	nRows=M.nRows;
	nCols=M.nCols;
	nStride=M.nStride;
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pBuff=(T *)pAlloc->Alloc((long)nRows*(long)nStride*sizeof(T));


	for(int i=0;i<nRows;i++)
		for(int j=0;j<nCols;j++)
			(*this)(i,j)=M(i,j);
}

template <typename T> Matrix<T>::Matrix() {
	nRows=0;
	nCols=0;
	pBuff=0;
	nStride=0;
}

template <typename T> Matrix<T>::Matrix(int nR,int nC,int Alignment) {

	pBuff=NULL;
	SetSize(nR,nC,Alignment);
}

template <typename T> Matrix<T>::Matrix(int nR,int nC,T *pData,int Alignment)  {
	pBuff=NULL;
	SetSize(nR,nC,Alignment);
	for(int i=0;i<nRows;i++)
		for(int j=0;j<nCols;j++)
			(*this)(i,j)=pData[i*nCols+j];
}


template <typename T> Matrix<T>::~Matrix() {
	if (pBuff!=NULL) {
		NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
		pAlloc->Free(pBuff,sizeof(T)*(long)nRows*(long)nStride);
	}
}

template <typename T> Matrix<T>& Matrix<T>::operator=(const Matrix<T> &M) {
	if (&M!=this) {
		NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
		if (pBuff!=NULL) {
			if (nRows!=M.nRows || nStride!=M.nStride  ) {
				pAlloc->Free(pBuff,sizeof(T)*(long)nRows*(long)nStride);
				pBuff=(T *)pAlloc->Alloc((long)M.nRows*(long)M.nStride*sizeof(T));
			}
		} else 	pBuff=(T *)pAlloc->Alloc((long)M.nRows*(long)M.nStride*sizeof(T));



		nRows=M.nRows;
		nCols=M.nCols;
		nStride=M.nStride;

		for(int i=0;i<nRows;i++)
			for(int j=0;j<nCols;j++)
				(*this)(i,j)=M(i,j);

	}
	return *this;
}


#endif /* UTIL_MATRIX_H_ */
