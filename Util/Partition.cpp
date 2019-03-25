

#include "Partition.h"
#include "FileException.h"
#include "Debug.h"

Partition::Partition(char *fname) {
	FILE *fd=fopen(fname,"rb");
	if (fd==NULL)
		throw FileException("Cannot open dataset");
	int nRows;
	if (fread(&nRows,sizeof(nRows),1,fd)!=1) {
		  throw FileException("fread failed");
		  fclose(fd);
	}
	ClsNums.SetSize(nRows);
	if (fread(ClsNums.GetData(),sizeof(int),nRows,fd)!=nRows) {
		fclose(fd);
		throw FileException("fread failed");
	}
	int MaxClass=0;
	for(int i=0;i<nRows;i++) {
		ClsNums[i]--;
		if (ClsNums[i]>MaxClass)
			MaxClass=ClsNums[i];
	}
	nClasses=MaxClass+1;
	DynamicArray<int> ClassCounts(nClasses);
	for(int i=0;i<nClasses;i++)
		ClassCounts[i]=0;

	for (int i=0;i<nRows;i++)
		ClassCounts[ClsNums[i]]++;

	ObjNums.SetSize(nClasses);
	for(int i=0;i<nClasses;i++)
		ObjNums[i].SetSize(ClassCounts[i]);


	for (int i=0;i<nClasses;i++) {
		int Cntr=0;
		DynamicArray<int> &Nums=ObjNums[i];
		for(int j=0;j<nRows;j++)
			if (ClsNums[j]==i)
				Nums[Cntr++]=j;
		ASSERT(Cntr==ClassCounts[i]);
	}
	printf("Partition consists of %d objects and %d classes\n",nRows,nClasses);
	for(int i=0;i<nClasses;i++) {
		printf("Class %d: %d Objects\n",i,ClassCounts[i]);
	}
	fclose(fd);
}

Partition::~Partition() {
	// TODO Auto-generated destructor stub
}

