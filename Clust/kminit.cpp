#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../Util/Rand.h"
#include "../Util/NumaDataset.h"
#include "../Util/StdOut.h"
#include "../Util/PrecisionTimer.h"
#include "KMeansInitializer.h"
#include "PlusPlusInitializer/PlusPlusInitializer.h"
#include "KMeansOrOrInitializer.h"
#include "KMeansReportWriter.h"

char *fname=NULL;
char *oname=NULL;
int ncl=3;
int verbosity=1;
int seed;

enum InitializerType {
	PLUSPLUS,
	FORGY,
	MINDIST,
	OROR,
} initializerType = FORGY;
double initializatorParam = -1;


void ExitUsage() {
	printf("Arguments filename -c nclusters -v verbosity -r randomseed\n");
	printf("-n initialization method parameter specifier\n");
	printf("-t initializer type, can be 'plusplus', 'forgy, 'oror' or 'mindist'\n");
	printf("-o file writes initial centroids to the file\n");
	printf("Defaults are: -c 3 -v 1 -r 0\n");
	exit(-1);
}

void InitRNGs() {
  	printf("Seeed of rng: %d\n",seed);
#ifdef _OPENMP
#pragma omp parallel
  	{
  		SRand(seed+omp_get_thread_num());
#pragma omp single
  		printf("OpenMP version with %d threads\n",omp_get_num_threads());
  	}
#else
	SRand(seed);
	printf("Single threaded version\n");
#endif
}

void DestroyRNGs() {
#pragma omp parallel
  	{
  		DelMTRand();
  	}
}



void ProcessArgs(int argc, char *argv[])
{
	int c;
	char *type;
	while ((c=getopt(argc,argv,"r:c:n:t:o:v:"))!=-1){
		switch(c) {
			case 'r':
					seed=atoi(optarg);
					if (seed<0) {
						printf("%s is an invalid value of -r option\n",optarg);
						ExitUsage();
					}
					break;
			case 'c':
					ncl=atoi(optarg);
					if (ncl<=0) {
						printf("%s is an invalid value of -c option\n",optarg);
						ExitUsage();
					}
					break;
			case 'n':
					initializatorParam = atof(optarg);
					break;
			case 't':
					type = optarg;
					if (strcmp(type, "plusplus") == 0) {
						initializerType = PLUSPLUS;
					} else if(strcmp(type, "mindist") == 0) {
						initializerType = MINDIST;
					} else if(strcmp(type, "oror") == 0) {
						initializerType = OROR;
					} else {
						initializerType = FORGY;
					}
					break;
			case 'o':
					oname=optarg;
					break;
			case 'v':
					verbosity=atoi(optarg);
					if (verbosity<0) {
						printf("%s is an invalid value of -v option\n",optarg);
						ExitUsage();
					}
					break;
			default: ExitUsage();
					break;
		}
	}
	if (optind<argc)
		fname=argv[optind];
	else
		ExitUsage();

}

KMeansInitializer* CreateKMInitializer(const InitializerType type, CentroidVector &CV, NumaDataset &D, int ncl, double initializatorParam) {
	switch(type) {
	case PLUSPLUS:
		printf("Kmeans++ initialization used\n");
		return new PlusPlusInitializer(D, CV, ncl);
	case MINDIST:
		printf("Minimum Distance initialization used\n");
		if (initializatorParam < 0) {
			printf("Minimum distance param cannot be lesser than zero! Set the -n option properly!\n");
			exit(-1);
		}
		printf("Number of min. distance candidates: %d\n", (int)initializatorParam);
		return new MinDistInitializer(D, CV, ncl, (int)initializatorParam);
	case OROR:
		if (initializatorParam<0)
			initializatorParam=1.0;
		printf("KMeans|| initialization used\n");
		printf("KMeans|| oversampling: %g\n",initializatorParam);
		return new KMeansOrOrInitializer(D,CV,ncl,initializatorParam);

	case FORGY:
	default:
		printf("Forgy initialization used\n");
		return new ForgyInitializer(D, CV, ncl);
	}
}


int main(int argc, char *argv[])
{
	printf("%s compilation time: %s %s\n",argv[0],__DATE__,__TIME__);

#ifdef BRANCH
	printf("Git branch: %s\n", BRANCH);
#endif
	StandardStdOut::Init();

#if defined(__INTEL_COMPILER)
	printf("Intel C++ compiler\n");
#elif defined (__PGI)
	printf("PGI C++ compiler\n");
#else
	printf("Other (probably g++) compiler\n");
#endif
	printf("sizeof(OPTFLOAT): %d\n",(int)sizeof(OPTFLOAT));
	printf("sizeof(EXPFLOAT): %d\n",(int)sizeof(EXPFLOAT));

	ProcessArgs(argc,argv);
  	InitRNGs();

	NumaDataset *pD=new NumaDataset;
	printf("NUMA optimizations on\n");

	printf("Loading dataset %s... ",fname);
	PrecisionTimer T(CLOCK_MONOTONIC);
	pD->Load(fname);
  	double extime=T.GetTimeDiff();

  	int nCols=pD->GetColCount();
  	int nRows=pD->GetRowCount();
  	printf("%d vectors, %d features\n",nRows,nCols);
	printf("Loading took %g seconds\n",extime);
	printf("Number of clusters: %d\n",ncl);
	CentroidVector CV(ncl,nCols);
	KMeansInitializer *pInit=CreateKMInitializer(initializerType,CV,*pD,ncl,initializatorParam);
	DynamicArray<OPTFLOAT> vec(nCols*ncl);
	T.Reset();
	pInit->Init(vec);
  	extime=T.GetTimeDiff();
	KMeansReportWriter Writer(*pD,ncl,NULL);
	if (oname==NULL)
		Writer.DumpClusters(vec);
	else {
		printf("Writing initial solution to file %s\n",oname);
		Writer.WriteCentroids(vec,oname);
	}
	printf("Initialization time: %g seconds\n",extime);

	delete pInit;
	delete pD;
	DestroyRNGs();
	return 0;
}
