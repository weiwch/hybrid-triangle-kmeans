#include "Debug.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

void DebugHang()
{
    fprintf(stderr,"Waiting for debugger (pid=%d)\n",getpid());
    fflush(stderr);
    while(1)
    	usleep(100000L);
}


void VerifyAssert(int x,char *pFile, int Line)
{
    if (x==0){
	fprintf(stderr,"Assertion failed [%s,line %d]\n",pFile,Line);
	DebugHang();
    }
}

#ifdef _DEBUG
void SwapStderr() {
	char name[128];
	snprintf(name,32,"Debug_%d.txt",getpid());
	VERIFY(freopen(name,"wt",stderr)!=NULL);
	ASSERT(stderr!=NULL);
}
#endif
