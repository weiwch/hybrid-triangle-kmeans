/*
 * MPIRank0StdOut.cpp
 *
 *  Created on: Oct 15, 2015
 *      Author: wkwedlo
 */

#include "MPIRank0StdOut.h"
#include <mpi.h>
#include <stdio.h>

void MPIRank0StdOut::Init() {
	if (instance!=NULL)
		delete instance;
	instance = new MPIRank0StdOut;
}

int MPIRank0StdOut::printf (const char *format, va_list arg) {
	int Rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	if (Rank==0) {
		int Ret=::vprintf(format,arg);
		fflush(stdout);
		return Ret;
	}
	return 0;
}
