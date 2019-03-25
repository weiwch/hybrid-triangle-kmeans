#include <cstdio>
#include <limits>
#include <unistd.h>
#include <math.h>
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
#include "KMAlgorithm.h"
#include "HamerlyKMA/HamerlyKMA.h"
#include "AnnulusKMA/AnnulusKMA.h"
#include "DrakeKMA/DrakeKMA.h"
#include "KMeansWithoutMSE/KMeansWithoutMSE.h"
#include "ElkanKMA/ElkanKMA.h"
#include "YinyangKMA.h"
#include "SameSizeKMA.h"
#include "KMeansReportWriter.h"



template class Array<float>;
template class DynamicArray<float>;
template class Array<int>;
template class DynamicArray<int>;
template class Array<Array<float>*>;
template class DynamicArray<Array<float>*>;

char *fname=NULL;
char *oname=NULL;
char *cname=NULL;
char *pname=NULL;
char *iname=NULL;
char *auxname=NULL;
int ncl=3;
int verbosity=1;
int seed;
double initializatorParam = -1;
bool sort;
double runtime=-1.0;
bool samesize=false;
bool global=false;
bool ward=false;
bool wardandkm=false;
char* yykmclusterer=NULL;
bool yykmlocal=false;
int yymkmfreclust=5;
double minrel=-1.0;
double hugepagespernode=1.0;
int groups=-1;
int b = ncl-1;
bool adaptiveDrake = false;
int maxiter=0;


enum AlgorithmType {
	ELKAN,
	HAMERLY,
	ANNULUS,
	DRAKE,
	NAIVE,
	YINYANG,
	YINYANGMOD
} algorithmType = NAIVE;

enum InitializerType {
	PLUSPLUS,
	FORGY,
	MINDIST,
	OROR,
	FILEI,
} initializerType = FORGY;

bool withoutMSE = false;

void ExitUsage() {
	printf("Arguments filename -c nclusters -v verbosity -r randomseed\n");
	printf("-n initialization method parameter specifier\n");
	printf("-a algorithm type, can be 'elkan', 'hamerly', 'annulus', 'drake', 'yinyang', 'yinyangmod' or 'naive'\n");
	printf("-t initializer type, can be 'plusplus', 'forgy' or 'mindist'\n");
	printf("-m enable no MSE version\n");
	printf("-i run the algorithm for a given number of seconds\n");
	printf("-g switches global k-means\n");
	printf("-w switches Ward's method\n");
	printf("-k switches Ward's method followed by k-means\n");
	printf("-o file writes quantized output to the file\n");
	printf("-O file writes classes of objects to the file\n");
	printf("-s switches same size clustering on\n");
	printf("-p writes (yinyang) or reads(the other algs) centroid vector permutation to/from file\n");
	printf("-N switches NUMA optimizations on (default is also on now)\n");
	printf("-C selects yinyangmod clustering method (default - kmeans)\n");
	printf("-L switches on local filtering in yinyangmod kmeans clustering\n");

	printf("-G sets the number of groups in yinyang kmeans\n");
	printf("-R sets the relative termination threshold\n");
	printf("(in case of SR -i sets the number of iterations)\n");
	printf("-d sets the number of bounds for DrakeKMA\n");
	printf("-D enable adaptive DrakeKMA bound calculation\n");
	printf("-b sets the first iteration for YinyangModKMA reclustering\n");
	printf("-I sets the maximum number of iterations");
	printf("Defaults are: -c 3 -v 1 -r 0\n");
	exit(-1);
}



void ProcessArgs(int argc, char *argv[])
{
	int c;
	char *type;
	while ((c=getopt(argc,argv,"LC:O:o:c:p:h:v:r:R:t:i:a:A:d:n:G:sgwkDmI:b:"))!=-1){
		switch(c) {
			case 'A':
					auxname=optarg;
					break;
			case 'L':
					yykmlocal=true;
					break;
			case 'C':
					yykmclusterer=optarg;
					break;

			case 'G':
					groups=atoi(optarg);
					break;
			case 'h':
					hugepagespernode=atof(optarg);
					break;
			case 'i':
					runtime=atof(optarg);
					if (runtime<=0) {
						printf("%s is an invalid value of -i option\n",optarg);
						ExitUsage();
					}
					break;
			case 'I':
					maxiter=atoi(optarg);
					if (maxiter<1) {
						kma_printf("%s is an invalid value of -I option\n",optarg);
						ExitUsage();
					}
					break;
			case 'b':
					yymkmfreclust=atoi(optarg);
					if (yymkmfreclust<1) {
						kma_printf("%s is an invalid value of -b option\n",optarg);
						ExitUsage();
					}
					break;
			case 'r':
					seed=atoi(optarg);
					if (seed<0) {
						printf("%s is an invalid value of -r option\n",optarg);
						ExitUsage();
					}
					break;
			case 'R':
					minrel=atof(optarg);
					if (minrel<0 || minrel >=1) {
						printf("%s is an invalid value of -R option\n",optarg);
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
			case 's':
					samesize=true;
					break;
			case 'g':
					global=true;
					break;
			case 'a':
					type = optarg;
					if (strcmp(type, "elkan") == 0) {
						algorithmType = ELKAN;
					} else if (strcmp(type, "hamerly") == 0) {
						algorithmType = HAMERLY;
					} else if (strcmp(type, "annulus") == 0) {
						algorithmType = ANNULUS;
					} else if (strcmp(type, "drake") == 0) {
						algorithmType = DRAKE;
					} else if (strcmp(type, "yinyang") == 0) {
						algorithmType = YINYANG;
					} else if (strcmp(type, "yinyangmod") == 0) {
						algorithmType = YINYANGMOD;
					} else 	{
						algorithmType = NAIVE;
					}
					break;
			case 'd':
					b = atoi(optarg);
					if(b < 0 || b > ncl-1) {
						printf("The 'd' param must be a value between 0 and %d.\n", ncl-1);
						ExitUsage();
					}
					break;
			case 'D':
					adaptiveDrake = true;
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
						initializerType = FILEI;
						iname=optarg;
					}
					break;
			case 'm':
					withoutMSE = true;
					break;
			case 'w':
					ward=true;
					break;
			case 'k':
					wardandkm=true;
					break;
			case 'o':
					oname=optarg;
					break;
			case 'p':
					pname=optarg;
					break;
			case 'O':
					cname=optarg;
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



YinyangClusterer *CreateYinyangClusterer() {
	if (yykmclusterer==NULL || !strcmp(yykmclusterer,"kmeans"))
		return new KMeansYinyangClusterer;
	if (!strcmp(yykmclusterer,"samesize"))
		return new SameSizeYinyangClusterer;
	ExitUsage();
}


KMeansInitializer* CreateKMInitializer(const InitializerType type, CentroidVector &CV, NumaDataset &D, int ncl, double initializatorParam) {
	switch(type) {
	case FILEI:
		printf("Reading initial centroids from file: %s\n",iname);
		if (access(iname,R_OK)) {
			printf("Cannot access this file!\n");
			exit(0);
		}
		return new FileInitializer(D,CV,ncl,iname);
		break;
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


double SameSizeKMeans(DynamicArray<OPTFLOAT> &vec,bool print,NumaDataset &D) {
	CentroidVector CV(ncl,D.GetColCount());
	KMeansInitializer *pInit = CreateKMInitializer(initializerType, CV, D, ncl, initializatorParam);
	pInit->Init(vec);
	SameSizeKMA KMA(CV,D);
	delete pInit;
	DynamicArray<int> Assignment(D.GetTotalRowCount());
	return KMA.TrainIterative(vec,Assignment,100,2);
}

double RunSR(DynamicArray<OPTFLOAT> &vec,KMeansInitializer *pInit,
			KMeansAlgorithm *pKMA,bool print) {

	printf("Staring k-means with stochastic relaxation\n");
	pInit->Init(vec);
	double BestFit=pKMA->RunKMeansWithSR(vec,print,runtime);
	printf("k-means with stochastic relaxation finished\n");
	return BestFit;
}


static int iter;
static double BestFit;

void RunKmeans(DynamicArray<OPTFLOAT> &vec,KMeansInitializer *pInit,KMeansAlgorithm *pKMA,StdDataset &D,int verbosity) {

	if (runtime <= 0) {
		PrecisionTimer TInit(CLOCK_MONOTONIC);
		pInit->Init(vec);
		double inittime=TInit.GetTimeDiff();
		kma_printf("Initialization time: %g seconds\n",inittime);
		YinyangKMA *pYYKMA=dynamic_cast<YinyangKMA*>(pKMA);
		if (pname!=NULL && pYYKMA==NULL) {
			printf("Permuting centroid vector using file %s\n",pname);
			CentroidVectorPermutation Perm(ncl);
			Perm.Read(pname);
			Perm.PermuteCentroidVector(vec,D.GetColCount());
		}

		PrecisionTimer TMonotonic(CLOCK_MONOTONIC),TThread(CLOCK_THREAD_CPUTIME_ID);


		KMeansWithoutMSE *noMSE = dynamic_cast<KMeansWithoutMSE*>(pKMA);

		if (noMSE != NULL && withoutMSE) {
			noMSE->RunKMeansWithoutMSE(vec, verbosity);
		} else if (noMSE == NULL && withoutMSE) {
			printf("Error: There is no MSE-less version of this algorithm.\n");
			return;
		} else {
			BestFit = pKMA->RunKMeans(vec, verbosity, minrel,maxiter);
		}

	  	double extime2_monotonic=TMonotonic.GetTimeDiff();
        double extime2_thread=TThread.GetTimeDiff();
        double diff=100.0*fabs((extime2_monotonic-extime2_thread)/extime2_thread);

		printf("%d total iterations\n",pKMA->GetIterCount());
		printf("k-means execution time: %g seconds\n",extime2_monotonic);
		printf("Highest absolute difference between thread 0 and real time: Process 0 Delta: %5.3f Relative: %1.3f%%\n",extime2_thread-extime2_monotonic,diff);
		printf("Wall clock resolution: %g seconds\n",TMonotonic.GetTick());
		printf("Thread clock resolution: %g seconds\n",TThread.GetTick());
		printf("k-means iteration time: %g seconds\n",extime2_monotonic/(double)pKMA->GetIterCount());
		if (pname!=NULL && pYYKMA!=NULL) {
			CentroidVectorPermutation &Perm=pYYKMA->GetPermutation();
			printf("Writing centroid vector permutation to %s\n",pname);
			Perm.Write(pname);
		}

	} else {
		PrecisionTimer T(CLOCK_MONOTONIC);
		DynamicArray<OPTFLOAT> tmpVec(vec.GetSize());
		BestFit=std::numeric_limits<OPTFLOAT>::max();
		while(true) {
			pInit->Init(tmpVec);
			double Fit=pKMA->RunKMeans(tmpVec,false,minrel,maxiter);
			iter++;
			double rtime=T.GetTimeDiff();
			if (Fit<BestFit) {
				BestFit=Fit;
				vec=tmpVec;
			}
			if (verbosity>0)
				printf("iter: %5d best: %6.8f time: %3.3f\n",iter,BestFit,rtime);
			if (rtime>=runtime) break;
		}
	}
}

KMeansAlgorithm* CreateKMAlgorithm(const AlgorithmType type, const bool withoutMSE, CentroidVector &CV, StdDataset &D, CentroidRepair *pRepair) {
	YinyangClusterer *pClust;
	switch (type) {

	case ANNULUS:
		if (withoutMSE) {
			printf("AnnulusKMA without MSE\n");
		} else {
			printf("AnnulusKMA\n");
		}
		return new AnnulusKMA(CV, D, pRepair);

	case DRAKE:
		if (withoutMSE) {
			printf("DrakeKMA without MSE\n");
		} else {
			printf("DrakeKMA\n");
		}
		if (adaptiveDrake) {
			printf("Adaptive DrakeKMA bounds enabled\n");
		} else {
			printf("Number of bounds set to b = %d\n",b);
		}
		return new DrakeKMA(CV, D, pRepair, b, adaptiveDrake);

	case YINYANG:
		if (withoutMSE) {
			printf("YinyangKMA without MSE\n");
		} else {
			printf("YinyangKMA\n");
		}
		return new YinyangKMA(CV, D, pRepair,groups,true);

	case ELKAN:
		if (withoutMSE) {
			printf("ElkanKMA without MSE\n");
		} else {
			printf("ElkanKMA\n");
		}
		return new ElkanKMA(CV, D, pRepair);

	case HAMERLY:
		if (withoutMSE) {
			printf("HamerlyKMA without MSE\n");
		} else {
			printf("HamerlyKMA\n");
		}
		return new HamerlyKMA(CV, D, pRepair);

	case NAIVE:
	default:
		if (withoutMSE) {
			printf("NaiveKMA without MSE\n");
		} else {
			printf("NaiveKMA\n");
		}		return new NaiveKMA(CV, D, pRepair);
	}
}



double LocalKMeans(DynamicArray<OPTFLOAT> &vec,int verbosity,NumaDataset &D,KMeansReportWriter *pKMRW) {
	CentroidVector CV(ncl,D.GetColCount());
	KMeansInitializer *pInit = CreateKMInitializer(initializerType, CV, D, ncl, initializatorParam);
	CentroidRandomRepair R(D,ncl);
	KMeansAlgorithm *pKMA = CreateKMAlgorithm(algorithmType, withoutMSE, CV, D, &R);
	double Best;

	pKMA->SetReportWriter(pKMRW);

	if (verbosity>4)
		pKMA->PrintNumaLocalityInfo();

	if (samesize) {
		if (runtime<=0) {
			printf("SR requires -i option\n");
			exit(-1);
		}
		Best=RunSR(vec,pInit,pKMA,verbosity>3);
	}else
		RunKmeans(vec,pInit,pKMA,D,verbosity);
	delete pInit;
	printf("%d k-means runs\n",iter);
	delete pKMA;
	return BestFit;
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

int main(int argc, char *argv[])
{

	PrecisionTimer T(CLOCK_MONOTONIC);

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
#ifdef NO_S
	printf("NO_S is enabled\n");
#endif
#ifdef NO_U
	printf("NO_U is enabled\n");
#endif
	printf("sizeof(OPTFLOAT): %d\n",(int)sizeof(OPTFLOAT));
	printf("sizeof(EXPFLOAT): %d\n",(int)sizeof(EXPFLOAT));

	ProcessArgs(argc,argv);
	if (minrel>0.0 && withoutMSE) {
		printf("-m and -R options are not compatible\n");
		StdOut::Destroy();
		return -1;
	}

	printf("Minimal number of huge pages (threshold) per NUMA Node: %1.2f\n",hugepagespernode);
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pAlloc->SetHugePageThreshold(hugepagespernode);
	NumaDataset *pD=new NumaDataset;
	printf("NUMA optimizations on\n");

	printf("Loading dataset %s... ",fname);
	PrecisionTimer TLoad(CLOCK_MONOTONIC);
	pD->Load(fname);
  	double extime2=TLoad.GetTimeDiff();
	DynamicArray<OPTFLOAT> vec;
	vec.SetSize(ncl*pD->GetColCount());
	printf("%d vectors, %d features\n",pD->GetRowCount(),pD->GetColCount());
	printf("Loading took %g seconds\n",extime2);
	//exit(0);
	if (verbosity>4) {
		printf("NUMA locality of dataset is %1.2f%%\n",pD->NumaLocality()*100.0);
	}
	printf("Number of clusters: %d\n",ncl);
	printf("First reclustering in YinyangMod at iteration: %d\n",yymkmfreclust);
	if (auxname)
		printf("Auxiliary experiment name: %s\n",auxname);
	KMeansReportWriter *pKMRW=NULL;
	if (oname!=NULL)
		pKMRW=new KMeansReportWriter(*pD,ncl,oname);

  	InitRNGs();
	double BestFit;
	bool print = verbosity>3 ? true : false;
	if (samesize)
		BestFit=SameSizeKMeans(vec,print,*pD);
	else
		BestFit=LocalKMeans(vec,verbosity,*pD,pKMRW);
	printf("Best MSE (SSE) %5.9f (%5.9f)\n",BestFit,BestFit*pD->GetRowCount());

	if (cname!=NULL) {
		CentroidVector CV(ncl,pD->GetColCount());
	}
	delete pD;
	if (pKMRW!=NULL)
		delete pKMRW;
	delete pAlloc;
	DestroyRNGs();
  	extime2=T.GetTimeDiff();
	printf("Total program execution time: %g seconds\n",extime2);
}
