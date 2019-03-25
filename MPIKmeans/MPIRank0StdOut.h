/*
 * MPIRank0StdOut.h
 *
 *  Created on: Oct 15, 2015
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_MPIRANK0STDOUT_H_
#define MPIKMEANS_MPIRANK0STDOUT_H_

#include "../Util/StdOut.h"

class MPIRank0StdOut: public StdOut {

public:
	static void Init();
	virtual int printf (const char *format, va_list arg);
};

#endif /* MPIKMEANS_MPIRANK0STDOUT_H_ */
