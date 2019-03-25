/*
 * LargeVector.h
 *
 *  Created on: Mar 18, 2015
 *      Author: wkwedlo
 */

#ifndef UTIL_LARGEVECTOR_H_
#define UTIL_LARGEVECTOR_H_

#include <sys/mman.h>
#include "Debug.h"
#include "NumaAlloc.h"


template <typename T> class LargeVector {

protected:
	virtual void AllocBuffer(int aSize);
	virtual void DestroyBuffer();

protected:
	T * __restrict__ pBuff;
	int Size;

public:
	T *GetData() {return pBuff;}
	const T *GetData() const {return pBuff;}
    int GetSize() const {return Size;}
	const T & operator[](int i) const;
    T & operator[](int i);
	LargeVector() {pBuff=NULL;Size=0;}
	void SetSize(int aSize);
	virtual ~LargeVector();
};


template <typename T> class ThreadPrivateVector : public LargeVector<T> {
	using LargeVector<T>::pBuff;
	using LargeVector<T>::Size;
protected:
	virtual void AllocBuffer(int aSize);
public:
	virtual ~ThreadPrivateVector() {}
};

template <typename T> LargeVector<T>::~LargeVector() {
	if (pBuff!=NULL)
		DestroyBuffer();
}


template <typename T> void  LargeVector<T>::DestroyBuffer() {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pAlloc->Free(pBuff,(long)Size*sizeof(T));
	pBuff=NULL;
	Size=0;
}

template <typename T> void LargeVector<T>::AllocBuffer(int aSize) {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pBuff=(T *)pAlloc->Alloc((long)aSize*sizeof(T));
	Size=aSize;
#pragma omp parallel for
	for(int i=0;i<Size;i++) {
		pBuff[i]=(T)0;
	}
}

template <typename T> void LargeVector<T>::SetSize(int aSize) {
	if (pBuff!=NULL)
		DestroyBuffer();
	AllocBuffer(aSize);
}

template <typename T> inline T &LargeVector<T>::operator[](int i) {
	ASSERT(i>=0);
	ASSERT(i<Size);
	return pBuff[i];
}

template <typename T> inline const T &LargeVector<T>::operator[](int i) const {
	ASSERT(i>=0);
	ASSERT(i<Size);
	return pBuff[i];
}

template <typename T> void ThreadPrivateVector<T>::AllocBuffer(int aSize) {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pBuff=(T *)pAlloc->Alloc((long)aSize*sizeof(T));
	Size=aSize;
	for(int i=0;i<Size;i++)
		pBuff[i]=(T)0;
}

#endif /* UTIL_LARGEVECTOR_H_ */
