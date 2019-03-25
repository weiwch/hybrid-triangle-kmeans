/*
 * StdOut.cpp
 *
 *  Created on: Oct 15, 2015
 *      Author: wkwedlo
 */

#include "StdOut.h"

#include <stdio.h>

StdOut *StdOut::instance;

void StdOut::Destroy() {
	if (instance!=NULL)
			delete instance;
}

void EmptyStdOut::Init() {
	if (instance!=NULL)
		delete instance;
	instance = new EmptyStdOut;
}


void StandardStdOut::Init() {
	if (instance!=NULL)
		delete instance;
	instance = new StandardStdOut;
}

int StandardStdOut::printf (const char *format, va_list arg) {
	int Ret=vprintf(format,arg);
	fflush(stdout);
	return Ret;
}

int kma_printf (const char *format, ...) {
	va_list ap;
	va_start(ap,format);
	StdOut *pInst=StdOut::GetInstance();
	if (pInst!=NULL)
		pInst->printf(format,ap);
	va_end(ap);
}

