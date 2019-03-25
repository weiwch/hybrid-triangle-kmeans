#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <sched.h>
#include <numa.h>


#include "../Util/PrecisionTimer.h"
#include "../Util/Rand.h"

#include "DistributedNumaDataset.h"
#include "MPIForgyInitializer.h"
#include "MPIKMeansOrOrInitializer.h"
#include "MPIKMAReducer.h"
#include "MPINaiveKMA.h"
#include "MPIYinyangKMA.h"
#include "MPIElkanKMA/MPIElkanKMA.h"
#include "MPIHamerlyKMA/MPIHamerlyKMA.h"
#include "MPIHamerlyKMA/MPIHamerlySmallestDistances.h"
#include "MPIElkanKMA/MPIElkanSmallestDistances.h"
#include "MPIAnnulusKMA/MPIAnnulusKMA.h"
#include "MPIDrakeKMA/MPIDrakeKMA.h"
#include "MPIRank0StdOut.h"
#include "MPICentroidRandomRepair.h"
#include "../Clust/CentroidVector.h"


double hugepagespernode=1.0;
char *fname=NULL;
char *reducer=NULL;
char *pname=NULL;
char *iname=NULL;
char *mpismallest=NULL;


int reducerbench=0;
int ncl=3;
int verbosity=1;
int repetitions=1;
int seed;
bool numa=true;
bool nomse=false;
double minrel=-1.0;
int groups=-1;
bool yykmcluster=true;
int maxiter=0;
int b = ncl - 1;
bool adaptiveDrake = false;
int reducerparam=-1;

enum AlgorithmType {
	ELKAN,
	HAMERLY,
	ANNULUS,
	YINYANG,
	DRAKE,
	NAIVE
} algorithmType = NAIVE;

enum InitializerType {
	FORGY,
	OROR,
	FILEI
} initializerType = FORGY;

void ExitUsage() {
	kma_printf("Arguments filename -a algorithm -c nclusters -v verbosity -r randomseed -n repetitions -R minrel -E reducername -F\n");
	kma_printf("-p writes (yinyang) or reads(the other algs) centroid vector permutation to/from file\n");
	kma_printf("Defaults are: -a naive -c 3 -v 1 -r 0 -n 1 -E simple\n");
	kma_printf("-I sets the maximum number of iterations");
	MPI_Finalize();
	exit(-1);
}


void ProcessArgs(int argc, char *argv[])
{
	int c;
	int Rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	while ((c=getopt(argc,argv,"b:CG:mn:r:c:v:E:e:a:h:R:p:M:I:d:Dt:"))!=-1){
		switch(c) {
			case 'C':
				yykmcluster=false;
				break;
			case 'b':
				reducerbench=atoi(optarg);
				break;
			case 'e':
				reducerparam=atoi(optarg);
				break;
			case 'G':
				groups=atoi(optarg);
				break;

			case 'R':
					minrel=atof(optarg);
					if (minrel<0 || minrel >=1) {
						printf("%s is an invalid value of -R option\n",optarg);
						ExitUsage();
					}
					break;

			case 'h': hugepagespernode=atof(optarg);
					  break;
			case 'm': nomse=true;
					break;
			case 'M': mpismallest=optarg;
					break;
			case 'E': reducer=optarg;
					break;
			case 'p':
					pname=optarg;
					break;
			case 'r':
					seed=atoi(optarg);
					if (seed<0) {
						kma_printf("%s is an invalid value of -r option\n",optarg);
						ExitUsage();
					}
					break;
			case 'n':
					repetitions=atoi(optarg);
					if (repetitions<1) {
						kma_printf("%s is an invalid value of -n option\n",optarg);
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
			case 'c':
					ncl=atoi(optarg);
					if (ncl<=0) {
						kma_printf("%s is an invalid value of -c option\n",optarg);
						ExitUsage();
					}
					break;
			case 'v':
					verbosity=atoi(optarg);
					break;
			case 'd':
					b = atoi(optarg);
					if(b < 0 || b > ncl-1) {
						kma_printf("The 'd' param must be a value between 0 and %d.\n", ncl - 1);
						ExitUsage();
					}
					break;
			case 'D':
					adaptiveDrake = true;
					break;
			case 'a':
					if (strcmp(optarg, "elkan") == 0) {
						algorithmType = ELKAN;
					} else if (strcmp(optarg, "hamerly") == 0) {
						algorithmType = HAMERLY;
					} else if (strcmp(optarg, "annulus") == 0) {
						algorithmType = ANNULUS;
					} else if (strcmp(optarg, "yinyang") == 0) {
						algorithmType = YINYANG;
					} else if (strcmp(optarg, "drake") == 0) {
						algorithmType = DRAKE;
					} else {
						algorithmType = NAIVE;
					}
					break;
			case 't':
					if(strcmp(optarg, "oror") == 0) {
						initializerType = OROR;
					} else {
						initializerType = FILEI;
						iname=optarg;
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

void PrintMPIandCPUInfo(int Rank) {
	char Name[MPI_MAX_PROCESSOR_NAME];
	int Len;
	MPI_Get_processor_name(Name,&Len);
#ifdef _OPENMP
	printf("For process %d  MPI_Get_processor_name returns %s\n",Rank,Name);
#pragma omp parallel
  	{
 		int CPU=sched_getcpu();
 		int Node=numa_node_of_cpu(CPU);
		printf("Process %d Thread %d CPU %d NUMA Node %d\n",Rank,omp_get_thread_num(),CPU,Node);
  	}
#else
  	{
		int CPU=sched_getcpu();
		int Node=numa_node_of_cpu(CPU);
		printf("For process %d  MPI_Get_processor_name returns %s, CPU %d NUMA Node %d\n",Rank,Name,CPU,Node);
  	}
#endif

}


void InitRNGs() {
#ifdef _OPENMP
#pragma omp parallel
  	{
  		SRand(seed+omp_get_thread_num());
  	}
#else
	SRand(seed);
#endif
}

void DestroyRNGs() {
#pragma omp parallel
  	{
  		DelMTRand();
  	}
}


HamerlySmallestDistances *CreateHamerlySmallestDistances(CentroidVector &CV) {
	if (mpismallest==NULL || !strcmp(mpismallest,"openmp")) {
		kma_printf("Using OpenMP smallest distances\n");
		return new HamerlyOpenMPSmallestDistances(CV);
	}
	if (!strcmp(mpismallest,"hybrid") || !strcmp(mpismallest,"hierarch")) {
		kma_printf("Using hierarch OpenMP/MPI smallest distances\n");
		return new MPIHamerlyHierarchSD(CV);
	}
	if (!strcmp(mpismallest,"hybrid2") || !strcmp(mpismallest,"crisscross")) {
		kma_printf("Using crisscrossed OpenMP/MPI smallest distances\n");
		return new MPIHamerlyCrisscrossSD(CV);
	}
	kma_printf("Unknown Hamerly smallest distances method: %s\n",mpismallest);
	MPI_Finalize();
	exit(-1);
}

ElkanSmallestDistances *CreateElkanSmallestDistances(CentroidVector &CV) {
	if (mpismallest==NULL || !strcmp(mpismallest,"openmp")) {
		kma_printf("Using OpenMP smallest distances\n");
		return new ElkanOpenMPSmallestDistances(CV);
	}
	if (!strcmp(mpismallest,"hybrid") || !strcmp(mpismallest,"hierarch")) {
		kma_printf("Using hierarch OpenMP/MPI smallest distances\n");
		return new MPIElkanHierarchSD(CV);
	}
	if (!strcmp(mpismallest,"hybrid2") || !strcmp(mpismallest,"criscross")) {
		kma_printf("Using crisscrossed OpenMP/MPI smallest distances\n");
		return new MPIElkanCrisscrossSD(CV);
	}
	kma_printf("Unknown Elkan smallest distances method: %s\n",mpismallest);
	MPI_Finalize();
	exit(-1);
}

KMeansInitializer *CreateMPIKMInitializer(InitializerType type,DistributedNumaDataset &Data,char *fname,CentroidVector &CV,int ncl) {
	switch(type) {
		case FILEI:
			kma_printf("Reading initial centroids from file: %s\n",iname);
			if (access(iname,R_OK)) {
				kma_printf("Cannot access this file!\n");
				MPI_Finalize();
				exit(0);
			}
			return new FileInitializer(Data,CV,ncl,iname);
			break;
		case OROR:
			return new MPIKMeansOrOrInitializer(Data,CV,ncl);
		default:
		case FORGY:
			return new MPIForgyInitializer(Data,fname,CV,ncl);
	}
}


KMeansAlgorithm* CreateMPIKMAlgorithm(const AlgorithmType type,
		const bool withoutMSE, CentroidVector &CV, DistributedNumaDataset &D,
		CentroidRepair *pRepair, MPIKMAReducer *pR,int yyGroups,bool yyCluster) {

	HamerlySmallestDistances *pDist;
	ElkanSmallestDistances *pEDist;
	switch (type) {
	case ELKAN:
		pEDist=CreateElkanSmallestDistances(CV);
		if (withoutMSE) {
			kma_printf("ElkanKMA without MSE\n");
		} else {
			kma_printf("ElkanKMA\n");
		}
		return new MPIElkanKMA(CV, D, pRepair, pR,pEDist);
	case HAMERLY:
		pDist=CreateHamerlySmallestDistances(CV);
		if (withoutMSE) {
			kma_printf("HamerlyKMA without MSE\n");
		} else {
			kma_printf("HamerlyKMA\n");
		}
		return new MPIHamerlyKMA(CV, D, pRepair, pR,pDist);
	case ANNULUS:
		pDist=CreateHamerlySmallestDistances(CV);
		if (withoutMSE) {
			kma_printf("AnnulusKMA without MSE\n");
		} else {
			kma_printf("AnnulusKMA\n");
		}
		return new MPIAnnulusKMA(CV, D, pRepair, pR,pDist);
	case YINYANG:
		if (withoutMSE) {
			kma_printf("YinyangKMA without MSE\n");
		} else {
			kma_printf("YinyangKMA\n");
		}
		return new MPIYinyangKMA(CV, D, pRepair,yyGroups,yyCluster,pR);
	case DRAKE:
		if (withoutMSE) {
			kma_printf("DrakeKMA without MSE\n");
		} else {
			kma_printf("DrakeKMA\n");
		}
		if (adaptiveDrake) {
			kma_printf("Adaptive MPIDrakeKMA bounds enabled\n");
		} else {
			kma_printf("Number of bounds set to b = %d\n",b);
		}
		return new MPIDrakeKMA(CV, D, pRepair, pR, b, adaptiveDrake);
	case NAIVE:
	default:
		if (withoutMSE)
			kma_printf("NaiveKMA without MSE\n");
		else
			kma_printf("NaiveKMA\n");
		return new MPINaiveKMA(CV, D, pRepair, pR);
	}
}

struct TimingInfo {
	double ThreadTime;
	double RealTime;
};


void PrintTimingInfo(double RealTime,double ThreadTime,int IterCount) {
	int Rank,Size;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);

	TimingInfo TI;
	TI.RealTime=RealTime;
	TI.ThreadTime=ThreadTime;
	DynamicArray<TimingInfo> TInfos(Size);

	MPI_Gather(&TI,sizeof(TI),MPI_BYTE,TInfos.GetData(),sizeof(TI),MPI_BYTE,0,MPI_COMM_WORLD);
	if (Rank==0) {
		if (verbosity>0) {
			for(int i=0;i<Size;i++)
				printf("Process%d real time: %5.3f thread 0 time: %5.3f Delta: %5.3f\n",i,TInfos[i].RealTime,TInfos[i].ThreadTime,
						TInfos[i].RealTime-TInfos[i].ThreadTime);
		}
		double MaxDelta=-1.0;
		int MaxProcess=-1;
		for(int i=0;i<Size;i++) {
			double Delta=std::abs(TInfos[i].RealTime-TInfos[i].ThreadTime);
			if (Delta>MaxDelta) {
				MaxDelta=Delta;
				MaxProcess=i;
			}

		}
		printf("%d total iterations\n",IterCount);
		printf("k-means execution time: %g seconds\n",RealTime);
		printf("k-means iteration time: %g seconds\n",RealTime/(double)IterCount);
		printf("Highest absolute difference between thread 0 and real time: Process %d Delta: %5.3f Relative: %1.3f%%\n",MaxProcess,MaxDelta,
				100.0*fabs(MaxDelta)/RealTime);
	}

}

double LocalKMeans(DynamicArray<OPTFLOAT> &vec,int verbosity,double MinRel,DistributedNumaDataset &D) {

	int nCols=D.GetColCount();

	CentroidVector CV(ncl,nCols);
	KMeansInitializer *pInit=CreateMPIKMInitializer(initializerType,D,fname,CV,ncl);
	MPIKMAReducer *pReducer=MPIKMAReducer::CreateReducer(reducer,ncl,nCols,reducerparam);
	MPICentroidRandomRepair R(D,ncl);

	if (pReducer==NULL) {
			kma_printf("Bad KMA reducer\n");
			MPI_Finalize();
			exit(0);
	}
	if (reducerbench>0) {
		pReducer->Benchmark(reducerbench);
		delete pReducer;
		return 0.0;
	}
	kma_printf("Using KMA reducer: %s\n",pReducer->GetName());
	kma_printf("Reduction portion: ");
	if (reducerparam==-1)
		kma_printf("all centroids\n");
	else
		kma_printf("%d centroids\n",reducerparam);
	KMeansAlgorithm *pKMA=CreateMPIKMAlgorithm(algorithmType, nomse, CV, D, &R, pReducer,groups,yykmcluster);
	double BestFit;

	if (verbosity>4)
		pKMA->PrintNumaLocalityInfo();

	for(int i=0;i<repetitions;i++) {
		InitRNGs();
		pKMA->ResetIterCount();
		pInit->Init(vec);


		MPIYinyangKMA *pYYKMA=dynamic_cast<MPIYinyangKMA*>(pKMA);
		if (pname!=NULL && pYYKMA==NULL) {
			kma_printf("Permuting centroid vector using file %s\n",pname);
			CentroidVectorPermutation Perm(ncl);
			Perm.Read(pname);
			Perm.PermuteCentroidVector(vec,D.GetColCount());
		}


		timespec tpstart_monotonic,tpend_monotonic;
        timespec tpstart_thread,tpend_thread;

		if (verbosity>0)
			kma_printf("Timing started\n");
		fflush(stdout);
		fsync(stdout->_fileno);
		MPI_Barrier(MPI_COMM_WORLD);

		PrecisionTimer TMonotonic(CLOCK_MONOTONIC_RAW), TThread(CLOCK_THREAD_CPUTIME_ID);
		if (nomse) {
			KMeansWithoutMSE *pKMANoMSE=dynamic_cast<KMeansWithoutMSE *> (pKMA);
			if (pKMANoMSE==NULL) {
				kma_printf("No MSE - less version of this algorithm");
				return 0.0;
			}
			pKMANoMSE->RunKMeansWithoutMSE(vec,verbosity);
			BestFit=0.0;
		} else
			BestFit = pKMA->RunKMeans(vec, verbosity, MinRel,maxiter);
		MPI_Barrier(MPI_COMM_WORLD);
		double extime2_monotonic=TMonotonic.GetTimeDiff();
        double extime2_thread=TThread.GetTimeDiff();
        if (verbosity>0)
			kma_printf("Timing finished\n");

		int Rank;
		MPI_Comm_rank(MPI_COMM_WORLD,&Rank);

		if (pname!=NULL && pYYKMA!=NULL && Rank==0) {
			CentroidVectorPermutation &Perm=pYYKMA->GetPermutation();
			printf("Writing centroid vector permutation to %s\n",pname);
			Perm.Write(pname);
		}

		PrintTimingInfo(extime2_monotonic,extime2_thread,pKMA->GetIterCount());

	}
	delete pKMA;
	delete pInit;
	delete pReducer;
	return BestFit;
}

void DoPs(int Rank) {
	char Buffer[1024];

	snprintf(Buffer,1024,"ps eax > ps%d.ps",Rank);
	system(Buffer);
}


int main(int argc, char *argv[])
{
	int Rank,Size;
	PrecisionTimer Total(CLOCK_MONOTONIC);

#ifdef _OPENMP
	int Provided;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&Provided);
#else
	MPI_Init(&argc,&argv);
#endif
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);

	ProcessArgs(argc,argv);
	if (verbosity>=0)
		MPIRank0StdOut::Init();
	else
		EmptyStdOut::Init();



	if (verbosity>1	)
		DoPs(Rank);

	if (minrel>0.0 && nomse) {
		kma_printf("-m and -R options are not compatible\n");
		StdOut::Destroy();
		MPI_Finalize();
		return -1;
	}

	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();

	kma_printf("%s compilation time: %s %s\n",argv[0],__DATE__,__TIME__);
#ifdef BRANCH
	kma_printf("Git branch: %s\n", BRANCH);
#endif
#if defined(__INTEL_COMPILER)
	kma_printf("Intel C++ compiler\n");
#elif defined (__PGI)
	kma_printf("PGI C++ compiler\n");
#else
	kma_printf("Other (probably g++) compiler\n");
#endif
	kma_printf("sizeof(OPTFLOAT): %d\n",(int)sizeof(OPTFLOAT));
	kma_printf("sizeof(EXPFLOAT): %d\n",(int)sizeof(EXPFLOAT));
	kma_printf("MPI application with %d processes\n",Size);
	kma_printf("Minimal number of huge pages (threshold) per NUMA Node: %1.2f\n",hugepagespernode);
	pAlloc->SetHugePageThreshold(hugepagespernode);
	InitRNGs();


	kma_printf("Loading dataset %s... ",fname);
	DistributedNumaDataset D;
	PrecisionTimer T(CLOCK_MONOTONIC);
	D.Load(fname);
	MPI_Barrier(MPI_COMM_WORLD);
	int nCols=D.GetColCount();
  	double extime2=T.GetTimeDiff();

	kma_printf("%d vectors, %d features\n",D.GetTotalRowCount(),D.GetColCount());
	kma_printf("Loading took %g seconds\n",extime2);
	kma_printf("Sum of data matrix: %f\n",D.GetTotalSum());
	kma_printf("Number of clusters: %d\n",ncl);


	if (verbosity>4) {
		printf("NUMA locality of dataset is %1.2f%%\n",D.NumaLocality()*100.0);
	}


	DynamicArray<OPTFLOAT> vec(ncl*nCols);
	LocalKMeans(vec,verbosity,minrel,D);
	if (verbosity>1)
		PrintMPIandCPUInfo(Rank);
	DestroyRNGs();
	delete pAlloc;
	MPI_Finalize();
	StdOut::Destroy();
  	extime2=(double)Total.GetTimeDiff();
	if (Rank==0)
		printf("Total program execution time: %g seconds\n",extime2);
}
