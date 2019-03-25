/*
 * KMeansReportWriter.h
 *
 *  Created on: Apr 24, 2017
 *      Author: wkwedlo
 */

#ifndef CLUST_KMEANSREPORTWRITER_H_
#define CLUST_KMEANSREPORTWRITER_H_

#include "CentroidVector.h"
#include "../Util/StdDataset.h"

class KMeansReportWriter {
	int ncols;
	int nclusters;
	const char *filestem;

	const StdDataset &Data;
	CentroidVector CV;

public:
	KMeansReportWriter(StdDataset &D,int ncl,const char *fstem);
	void DumpClusters(Array<OPTFLOAT> &v);
	void DumpCentroids(Array<OPTFLOAT> &v);
	void QuantizeDataset(Array<OPTFLOAT> &v,const char *fname);
	void WriteCentroids(Array<OPTFLOAT> &v,const char *fname);
	void WriteClasses(Array<OPTFLOAT> &v,const char *fname);

	void IterationReport(Array<OPTFLOAT> &v,int i);

	virtual ~KMeansReportWriter() {}

};

#endif /* CLUST_KMEANSREPORTWRITER_H_ */
