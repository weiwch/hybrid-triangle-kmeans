/*
 * StdOut.h
 *
 *  Created on: Oct 15, 2015
 *      Author: wkwedlo
 */

#ifndef UTIL_STDOUT_H_
#define UTIL_STDOUT_H_

#include <stdarg.h>

class StdOut {

protected:
	static StdOut *instance;


public:
	virtual ~StdOut() {}
	virtual int printf (const char *format, va_list arg)=0;
	static StdOut * GetInstance() {return instance;}
	static void Destroy();
};



class EmptyStdOut : public StdOut {


public:
	static void Init();
	virtual int printf (const char *format, va_list arg) {return 0;}
};


class StandardStdOut : public StdOut {

public:
	static void Init();
	virtual int printf (const char *format, va_list arg);
};

int kma_printf(const char *format,...);

#endif /* UTIL_STDOUT_H_ */
