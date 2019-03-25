/*
 * PrecisionTimer.cpp
 *
 *  Created on: Apr 13, 2017
 *      Author: wkwedlo
 */

#include "PrecisionTimer.h"


static void timespec_diff(timespec *start, timespec *stop,
                   timespec *result)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }

    return;
}


PrecisionTimer::PrecisionTimer(clockid_t id) {
	clk_id=id;
	Reset();
}

PrecisionTimer::~PrecisionTimer() {
}

void PrecisionTimer::Reset() {
	clock_gettime(clk_id,&tp_start);
}

double PrecisionTimer::GetTimeDiff() {
	clock_gettime(clk_id,&tp_end);
	return TimeDifference();
}

double PrecisionTimer::GetTimeDiffAndReset() {
	clock_gettime(clk_id,&tp_end);
	double Diff=TimeDifference();
	tp_start=tp_end;
	return Diff;
}

double PrecisionTimer::TimeDifference() {
	timespec diff;
	timespec_diff(&tp_start,&tp_end,&diff);
	return (double)diff.tv_sec+1e-9*(double)diff.tv_nsec;

}

double PrecisionTimer::GetTick() {
	timespec tick;
	clock_getres(clk_id,&tick);
	return (double)tick.tv_sec+1e-9*(double)tick.tv_nsec;
}
