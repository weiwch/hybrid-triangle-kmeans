#include <limits>
#include <stdio.h>
#include "KMeansInitializer.h"
#include "../Util/Rand.h"
#include "../Util/FileException.h"

KMeansInitializer::KMeansInitializer(StdDataset &D,CentroidVector &aCV,int cl): 
					Data(D),CV(aCV),nclusters(cl) {
	ncols=Data.GetColCount();
}  




void KMeansInitializer::RepairCentroid(Array<OPTFLOAT> &vec,int clnum) {
	//printf("*");
	//fflush(stdout);
	
		DataRow &row=Data.GetRow((int)(Rand()*Data.GetRowCount()));
		for(int j=ncols*clnum,k=0;j<ncols*(clnum+1);j++,k++)
			vec[j]=row[k];
}



ForgyInitializer::ForgyInitializer(StdDataset &D,CentroidVector &aCV,int cl) : KMeansInitializer(D,aCV,cl) {
}

//#define TRACE_OBJNUMS
//#define TRACE_CENTROIDS


void ForgyInitializer::Init(Array<OPTFLOAT> &v) {
	ASSERT(v.GetSize()==ncols*nclusters);
	//printf("Using Forgy Initializer\n");
	for(int i=0;i<nclusters;i++) {
		int ObjNum=(int)(Rand()*Data.GetRowCount());
#ifdef TRACE_OBJNUMS
		TRACE1("Obj %d selected as initial centroid\n",ObjNum);
#endif
		DataRow &row=Data.GetRow(ObjNum);
		for(int j=0;j<ncols;j++)
			v[i*ncols+j]=row[j];
	}
#ifdef TRACE_CENTROIDS
	for(int i=0;i<nclusters*ncols;i++)
			TRACE2("v[%d]=%f\n",i,v[i]);
#endif
}




RandomInitializer::RandomInitializer(StdDataset &D,CentroidVector &aCV,int cl) : KMeansInitializer(D,aCV,cl) {
	PartCounts.SetSize(nclusters);
}

void RandomInitializer::Init(Array<OPTFLOAT> &v) {
	int nRows=Data.GetRowCount();
	
	for (int i=0;i<nclusters;i++)
		PartCounts[i]=0;
	
	for (int i=0;i<nclusters*ncols;i++)
		v[i]=(OPTFLOAT)0;
	
	for(int i=0;i<nRows;i++) {
		const DataRow &row=Data.GetRow(i);
		int Part=(int)(Rand()*nclusters);
		PartCounts[Part]++;
		for(int j=0;j<ncols;j++)
			v[Part*ncols+j]+=row[j];
	}
	for (int i=0;i<nclusters;i++) {
		OPTFLOAT f=(OPTFLOAT)1.0/(OPTFLOAT)PartCounts[i];
		TRACE2("Cluster%d f=%5.3f\n",i,f);
		for (int j=0;j<ncols;j++)
			v[i*ncols+j]*=f;
	}
}



MinDistInitializer::MinDistInitializer(StdDataset &D,CentroidVector &aCV,int cl,int Tr) : KMeansInitializer(D,aCV,cl) {
	Trials=Tr;
}


void MinDistInitializer::Init(Array<OPTFLOAT> &v) {
    ASSERT(v.GetSize()==nclusters*ncols);
	DynamicArray<int> Nums(Trials);
	DynamicArray<OPTFLOAT> Center(ncols);
	int nRows=Data.GetRowCount();
	
	for(int i=0;i<ncols;i++) {
		Center[i]=0.0;
	}
	for(int i=0;i<nRows;i++) {
		const DataRow &Row=Data.GetRow(i);
		for(int j=0;j<ncols;j++) {
			Center[j]+=Row[j];
		}
	}
	OPTFLOAT f=(OPTFLOAT)1.0/(OPTFLOAT)nRows;
	
	for(int i=0;i<ncols;i++) {
		Center[i]*=f;
	}
	
	OPTFLOAT MinDist=std::numeric_limits<double>::max();	
	for(int i=0;i<Trials;i++) {
		DataRow &Row=Data.GetRow((int)(Rand()*Data.GetRowCount()));
		OPTFLOAT Dist=CV.SquaredDistance(0,Center,Row);
		if (Dist<MinDist) {
			MinDist=Dist;
			for(int j=0;j<ncols;j++)
				v[j]=Row[j];
		}
	}
	
	
	/*DataRow &rfirst=Data.GetRow((int)(Rand()*nRows));
	for (int i=0;i<ncols;i++)
			v[i]=rfirst[i];*/
	for(int i=1;i<nclusters;i++) {
			for(int j=0;j<Trials;j++)
				Nums[j]=(int)(Rand()*Data.GetRowCount());
			double MaxDist=0;
			int Maxj=0;
			for(int j=0;j<Trials;j++) {
				DataRow &row=Data.GetRow(Nums[j]);
				double MinDist=std::numeric_limits<double>::max();
				for(int k=0;k<i;k++) {
					double SD=CV.SquaredDistance(k,v,row);
					//printf("v%d - c%d : %5.3f\n",j,k,SD);
					if (SD < MinDist)
						MinDist=SD;
				}
				//printf("v%d mindist %5.3f\n",j,MinDist);
				if (MinDist>MaxDist) {
					MaxDist=MinDist;
					Maxj=j;
				}
			}
			//printf("v%d selected\n",Maxj);
			DataRow &row=Data.GetRow(Nums[Maxj]);
			for(int j=ncols*i,k=0;j<ncols*(i+1);j++,k++)
				v[j]=row[k];
			}
	//exit(-1);
}

FileInitializer::FileInitializer(StdDataset &D,CentroidVector &aCV,int cl,const char *fname) : KMeansInitializer(D,aCV,cl) {
	FILE *pF=fopen(fname,"rb");
	if (pF==NULL)
		throw FileException("Cannot open file with centroid coordinates");
	int nCols,nRows;
	fread(&nRows,1,sizeof(nRows),pF);
	fread(&nCols,1,sizeof(nCols),pF);
	if (nCols!=ncols)
		throw FileException("Number of columns in dataset and centroid file do not match");
	if (nRows!=nclusters)
		throw FileException("Number of clusters in centroid file and -c value do not match");
	vec.SetSize(ncols*nclusters);
	if (fread(vec.GetData(),sizeof(OPTFLOAT),nclusters*ncols,pF)!=nclusters*ncols)
		throw FileException("fread from centroid file failed");
}

void FileInitializer::Init(Array<OPTFLOAT> &v) {
	v=vec;
}
