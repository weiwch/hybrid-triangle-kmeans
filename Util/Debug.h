


#ifndef __DEBUG_H
#define __DEBUG_H

#include <stdio.h>
void VerifyAssert(int x,char *pFile,int Line);
void DebugHang();


#ifdef _DEBUG
#define ASSERT(x) VerifyAssert(x,(char *)__FILE__,__LINE__)
#define VERIFY(x) VerifyAssert(x,(char *)__FILE__,__LINE__)
void SwapStderr();
#else
#define ASSERT(x)
#define VERIFY(x) (x)
static inline void SwapStderr() {;}
#endif


#if (defined _DEBUG && !defined _NOTRACE) || defined _TRACE
#define TRACE0(x)			(fprintf(stderr,(x)),fflush(stderr))
#define TRACE1(x,p1)		(fprintf(stderr,(x),(p1)),fflush(stderr))	
#define TRACE2(x,p1,p2)		(fprintf(stderr,(x),(p1),(p2)),fflush(stderr))
#define TRACE3(x,p1,p2,p3)	(fprintf(stderr,(x),(p1),(p2),(p3)),fflush(stderr))
#define TRACE4(x,p1,p2,p3,p4)	(fprintf(stderr,(x),(p1),(p2),(p3),(p4)),fflush(stderr))

#else
#define TRACE0(x)
#define TRACE1(x,p1)
#define TRACE2(x,p1,p2)
#define TRACE3(x,p1,p2,p3)
#define TRACE4(x,p1,p2,p3,p4)
#endif

#endif
