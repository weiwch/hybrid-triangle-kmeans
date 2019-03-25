// -*- C++ -*-


#ifndef __ARRAY_H

#ifndef OPTFLOAT
#define  OPTFLOAT float
#endif

#include "Debug.h"
#include <cstring>
#include <cstdlib>
#include <new>


#define __ARRAY_GROW_BY 16;
#define __ARRAY_H


//#define _DEBUG_FUNCTION_CALLS

#ifdef _DEBUG
#define _RANGE_CHECK_
#else
#undef  _DEBUG_FUNCTION_CALLS
#endif



void CheckArrayRange(int i,int Size);
void NumaScan(char *ptr,int nBytes);


// Supress inline expansion if the range check is enabled

#ifdef _RANGE_CHECK_
#define ARR_INLINE
#else
#define ARR_INLINE  inline
#endif





/// Template for a C-style array, without memory management !!!

/** This is an MFC CArray/STL vector style template of the array class with
    operator []. This class is provided without any memory management - the user has 
    to provide a buffer in a constructor and has to free the buffer after the Array
    had been destroyed.
    
    If you need memory management then use  DynamicArray.
    
    If the macro _DEBUG is defined all operations use range checks.
*/

template <typename T> class Array {
protected:
	/// array buffer
	T * __restrict__ pBuff;
	/// array size used in index range checks
	int Size;
public:
    Array(T *pB,int S);
    /// Constructor for derrived classes only
    Array();
    Array &operator=(Array &arr);
    virtual ~Array<T>() {}
    /// array index operator
    T & operator[](int Idx);
    /// array index operator for read access
    const T & operator[](int Idx) const;
    T *operator+(int Idx);
    /// access to buffer
    T *GetData() const {return pBuff;}
    /// conversion to pointer
    int GetSize() const {return Size;}
    /// Check distribution of memory into NUMA nodes

    void ScanNUMANodes() { NumaScan((char *)pBuff,Size*sizeof(T));}
};

/**
 Main constructor for the Array template
 \param pB - buffer for the array, must be valid during the object lifetime
 \param S - size of the buffer/array
 */

template <typename T> Array<T>::Array(T *pB,int S) 
: pBuff(pB),
  Size(S)
{
//	pBuff=pB;
//	Size=S;
}

template <typename T> Array<T>::Array()
: pBuff(NULL),
  Size(0)
{
//	pBuff=NULL;
//	Size=0;
}

template <typename T> ARR_INLINE T& Array<T>::operator[](int Idx)
{
#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,Size);
#endif
    return pBuff[Idx];
}

template <typename T> ARR_INLINE const T& Array<T>::operator[](int Idx) const
{
#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,Size);
#endif
    return pBuff[Idx];
}

template <typename T> ARR_INLINE T* Array<T>::operator+(int Idx)
{
#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,Size);
#endif
    return pBuff+Idx;
}

template <typename T> Array<T> & Array<T>::operator=(Array<T> &arr) 
{
	ASSERT(Size==arr.Size);
	for(int i=0;i<Size;i++)
		pBuff[i]=arr[i];
	return *this;
}

/// Template for a C-style array, with dynamic memory management

/** The DynamicArray template class extends Array with dynamic memory 
 * (allocated from heap)
 * management. This class is very similiar to MFC CArray/std::vector
   You can use SetSize to change the size of the array. Destructor
   frees all the memory.
 */
template <typename T> class DynamicArray : public Array<T>
{

protected:
	using Array<T>::pBuff;
	using Array<T>::Size;
    int BufferSize;
	
    void Grow(int aSize);
    void Shrink(int aSize);
    void ConstructBuffer(int aSize);
    void DestroyBuffer();
    void NewBuffer(int aSize);
    void ShrinkIfNeeded(int aSize);
    void GrowIfNeeded(int aSize);

	static void ConstrElems(T* pBuffer, int Count);
	static void DestrElems(T* pBuffer,int Count);

 public:
    /// Constructs array with initial size = 0
    DynamicArray() : BufferSize(0) { /*BufferSize=0;*/}
    /// Constructs array with initial size = aSize
    DynamicArray(int aSize);
    DynamicArray(const Array<T> &A);
    DynamicArray(const DynamicArray<T> &A);

    /// Destructor frees all the memory
    ~DynamicArray();
    /// Changes the size of the array, preserving data if possible
    void SetSize(int aSize,int GrowBy=-1);
    /// Appends Src to the end of array
    int Append(Array<T> &Src);
	/// Appends Elem to the end of array
    void Add(const T &Elem);
	/// Removes Count elements starting from position Idx    
    void RemoveAt(int Idx,int Count=1);
  
    void InsertAt(int Idx, T &Elem,int Count=1);
    DynamicArray &operator=(const Array<T> &A1);

};


template<typename T> void DynamicArray<T>::ConstrElems(T* pBuffer, int Count)
{

    memset((void*)pBuffer,0,Count*sizeof(T));

    for (; Count--; pBuffer++)
	new ((void *)pBuffer) T;
}

template<typename T> void DynamicArray<T>::DestrElems(T* pBuffer,int Count)
{
    for (; Count--; pBuffer++)
	pBuffer->~T();
}



template <typename T> void DynamicArray<T>::ConstructBuffer(int aSize)
{
    pBuff=(T *)malloc(sizeof(T)*aSize);
    ConstrElems(pBuff,aSize);
}

template <typename T> void DynamicArray<T>::NewBuffer(int aSize)
{
    T * pTmp=(T *)::malloc(sizeof(T)*aSize);
    memmove(pTmp,pBuff,Size*sizeof(T));
    free(pBuff);
    pBuff=pTmp;
    BufferSize=aSize;
}

template <typename T> void DynamicArray<T>::DestroyBuffer()
{
    DestrElems(pBuff,Size);
    free(pBuff);
    pBuff=NULL;
    Size=BufferSize=0;
}

template <typename T> DynamicArray<T>::DynamicArray(int aSize)
: BufferSize(aSize)
{
//    BufferSize=Size=aSize;
    ConstructBuffer(BufferSize);
    Size = BufferSize;
	   
}

template <typename T> DynamicArray<T>::DynamicArray(const Array<T> & A)
{
    BufferSize=Size=A.GetSize();
    T *pOldBuff=A.GetData();
    ConstructBuffer(Size);    
    for(int i=0;i<Size;i++)
		pBuff[i]=pOldBuff[i];
}

template <typename T> DynamicArray<T>::DynamicArray(const DynamicArray<T> & A)
{
    BufferSize=Size=A.GetSize();
    T *pOldBuff=A.GetData();
    ConstructBuffer(Size);
    for(int i=0;i<Size;i++)
		pBuff[i]=pOldBuff[i];
}

template <typename T> DynamicArray<T>::~DynamicArray()
{
	ASSERT(BufferSize>=0);
	if (pBuff!=NULL)
		DestroyBuffer();
}


template <typename T> DynamicArray<T> & DynamicArray<T>::operator=(const Array<T> &A)
{

#ifdef _DEBUG_FUNCTION_CALLS
    fprintf(stderr,"Array::operator= ");
    fprintf(stderr,"Size %d BufferSize %d\n",Size,BufferSize);
#endif

    SetSize(A.GetSize());
    for(int i=0;i<Size;i++)
	pBuff[i]=A[i];
    return *this;
}


template <typename T> int DynamicArray<T>::Append(Array<T> &Src)
{
    int Ret=Size;
    T *pSrcBuff=Src.GetData();
    SetSize(Ret+Src.GetSize());
    for(int i=0;i<Src.GetSize();i++)
		pBuff[Ret+i]=pSrcBuff[i];
    return Ret;
}

template <typename T> void DynamicArray<T>::SetSize(int aSize,int GrowBy)
{
    ASSERT(BufferSize>=0);
    if (pBuff==NULL){
	Size=BufferSize=aSize;
	if (aSize!=0)
	    ConstructBuffer(aSize);
	return;
    }
	
    if (aSize==0) {
		DestroyBuffer();
		return;
    }


    if (aSize<Size)
		Shrink(aSize);
    if (aSize>Size) 
		Grow(aSize);
}


template <typename T> void DynamicArray<T>::GrowIfNeeded(int aSize)
{
    if (aSize>BufferSize) {
	int NewBufferSize = (int)(aSize*2);

#ifdef _DEBUG_FUNCTION_CALLS
	fprintf(stderr,"Buffer grown from %d to %d\n",BufferSize,NewBufferSize);
#endif
	ASSERT(NewBufferSize>=aSize);
	NewBuffer(NewBufferSize);
    }
}

template <typename T> void DynamicArray<T>::Grow(int aSize)
{

#ifdef _DEBUG_FUNCTION_CALLS
    fprintf(stderr,"Array::Grow, aSize: %d",aSize);
    fprintf(stderr,"Size %d BufferSize %d\n",Size,BufferSize);
#endif
    GrowIfNeeded(aSize);
    ConstrElems(pBuff+Size,aSize-Size);
    Size=aSize;
}

template <typename T> void DynamicArray<T>::ShrinkIfNeeded(int aSize)
{
    if (BufferSize>aSize*2){		
	int NewBufferSize = (int)(aSize*1.5);
#ifdef _DEBUG_FUNCTION_CALLS
	fprintf(stderr,"Buffer shrunk from %d to %d\n",BufferSize,NewBufferSize);
#endif
	ASSERT(NewBufferSize>=aSize);
	NewBuffer(NewBufferSize);
    }
}

template <typename T> void DynamicArray<T>::Shrink(int aSize)
{

#ifdef _DEBUG_FUNCTION_CALLS
    fprintf(stderr,"Array::Shrink, aSize: %d",aSize);
    fprintf(stderr,"Size %d BufferSize %d\n",Size,BufferSize);
#endif

    if (aSize==0){
	DestroyBuffer();
	return;
    }
    DestrElems(pBuff+aSize,Size-aSize);
    Size=aSize;
    ShrinkIfNeeded(Size);
}

template <typename T> ARR_INLINE void DynamicArray<T>::Add(const T &Elem)
{

#ifdef _DEBUG_FUNCTION_CALLS
    fprintf(stderr,"Array::Add");
    fprintf(stderr,"Size %d BufferSize %d\n",Size,BufferSize);
#endif

    SetSize(Size+1);
    pBuff[Size-1]=Elem;
}


template <typename T> void DynamicArray<T>::RemoveAt(int Idx,int Count)
{

#ifdef _DEBUG_FUNCTION_CALLS
    fprintf(stderr,"Array::RemoveAt Idx: %d",Idx);
    fprintf(stderr," Size %d BufferSize %d\n",Size,BufferSize);
#endif


#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,Size);
#endif
	
    int MoveCount = Size - (Idx + Count);
    DestrElems(pBuff+Idx,Count);
    memmove(pBuff+Idx,pBuff+Idx+Count,MoveCount*sizeof(T));
    Size-=Count;
    ShrinkIfNeeded(Size);
}


// Important difference (?) from MFC:
// insert only at indices in range 0,..,GetSize()

template <typename T> void DynamicArray<T>::InsertAt(int Idx, T &Elem,int Count)
{

#ifdef _DEBUG_FUNCTION_CALLS
    fprintf(stderr,"Array::InsertAt Idx: %d",Idx);
    fprintf(stderr,"Size %d BufferSize %d\n",Size,BufferSize);
#endif

#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,Size);
#endif

    int MoveCount = Size -Idx;
    GrowIfNeeded(Size+Count);
    if (MoveCount>0)
	memmove(pBuff+Idx+Count,pBuff+Idx,MoveCount*sizeof(T));
    ConstrElems(pBuff+Idx,Count);
    for(int i=0;i<Count;i++)
	pBuff[Idx+i]=Elem;
    Size+=Count;
}
	

#endif // __ARRAY_H
