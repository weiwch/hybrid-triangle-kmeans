/*
 * PrecisionTimer.h
 *
 *  Created on: Apr 13, 2017
 *      Author: wkwedlo
 */

#ifndef UTIL_PRECISIONTIMER_H_
#define UTIL_PRECISIONTIMER_H_
#include <time.h>

class PrecisionTimer {

	timespec tp_start,tp_end;
	clockid_t clk_id;

	double TimeDifference();

public:
	void Reset();

	double GetTimeDiff();
	double GetTimeDiffAndReset();
	double GetTick();
	PrecisionTimer(clockid_t id);
	virtual ~PrecisionTimer();
};

#endif /* UTIL_PRECISIONTIMER_H_ */
