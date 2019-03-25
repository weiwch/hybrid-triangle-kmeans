/*
 * Partition.h
 *
 *  Created on: Nov 12, 2012
 *      Author: wkwedlo
 */

#ifndef PARTITION_H_
#define PARTITION_H_

#include "Debug.h"
#include "Array.h"

class Partition {

	DynamicArray <int> ClsNums;
	DynamicArray < DynamicArray<int> > ObjNums;
	int nClasses;

public:
	Partition(char *fname);
	int GetClassCount() {return nClasses;}
	int GetObjCount(int Class) const  {return ObjNums[Class].GetSize();}
	int GetObjNum(int Class,int Obj) const {return ObjNums[Class][Obj];}
	int GetClass(int Obj) const {return ClsNums[Obj];}
	virtual ~Partition();
};

#endif /* PARTITION_H_ */
