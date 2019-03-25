#include "KMAlgorithm.h"
#include <limits>
#include <math.h>
#include <sys/times.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "../Util/OpenMP.h"
#include "../Util/StdOut.h"
#include "../Util/PrecisionTimer.h"
#include "../Util/Rand.h"


int KMeansAlgorithm::LastIter;
double KMeansAlgorithm::LastTime;



KMeansAlgorithm::KMeansAlgorithm(CentroidVector &aCV,StdDataset &D,CentroidRepair *pR) : CV(aCV),Data(D)
{
	nclusters=CV.GetNClusters();
	ncols=CV.GetNCols();
	pRepair=pR;
	IterCount=0;
	pReportWriter=NULL;
}

void KMeansAlgorithm::PeturbVector(Array<OPTFLOAT> &vec, double rtime,double MaxTime) {
	for(int i=0;i<nclusters;i++) {
		for(int j=0;j<ncols;j++) {
			double Std=Data.GetStdDev(j);
			double Scale=pow(1.0-rtime/MaxTime,4.0);
			//double Scale=pow(0.98,Iter);
			vec[i*ncols+j]+=((Rand()-0.5)*Std*sqrt(Scale));
			//vec[i*ncols+j]+=(RandNorm()*0.33*Std*sqrt(Scale));
		}
	}
}

void KMeansAlgorithm::PrintIterInfo(int i, double BestFit, double Rel,double distanceCalculationsRatio, double iterTime) {
	kma_printf("i: %d fit: %g rel: %g, dcr: %g, itime: %g seconds\n",i,BestFit,Rel,distanceCalculationsRatio, iterTime);
}

void KMeansAlgorithm::PrintAvgAvoidance(double distanceCalculationsRatio) {
    kma_printf("Avg distance calculations ratio: %g\n",distanceCalculationsRatio);
}


void signal_handler(int sig) {
	fprintf(stderr,"Got signal %d, Last iteration: %d at time %g seconds\n",sig,KMeansAlgorithm::GetLastIter(),KMeansAlgorithm::GetLastTime());
	fflush(stderr);
}


double KMeansAlgorithm::RunKMeans(Array<OPTFLOAT> &vec, int verbosity,double MinRel,int MaxIter)
{
	ASSERT(vec.GetSize()==ncols*nclusters);
	volatile double diff;
	int i;
        long distanceCount = 0;
        double distanceCalculationsRatio=0;
        double distanceCalculationsRatioSum=0;
	double BestFit=std::numeric_limits<double>::max();
	PrecisionTimer T(CLOCK_MONOTONIC);
	InitDataStructures(vec);
	if (verbosity>4 && pReportWriter!=NULL)
		pReportWriter->IterationReport(vec,0);

	struct sigaction new_action;
	new_action.sa_flags=0;
	new_action.sa_handler=signal_handler;
	sigemptyset(&new_action.sa_mask);
	sigaction(SIGTERM,&new_action,NULL);

	for(i=1;;i++) {
		distanceCount=0;
		timespec tpend_monotonic;

		double Fit=ComputeMSEAndCorrect(vec, &distanceCount);
	  	double iterTime=T.GetTimeDiff();

	  	LastIter=i;
	  	LastTime=iterTime;

        distanceCalculationsRatio = ((double) (distanceCount / (double) (nclusters * Data.GetTotalRowCount()))) * 100;
        distanceCalculationsRatioSum += distanceCalculationsRatio;
		double Rel=(BestFit-Fit)/BestFit;
		if (Fit<nextafter(Fit,BestFit) ) {
			BestFit=Fit;
			if (verbosity>3)
				PrintIterInfo(i,BestFit,Rel,distanceCalculationsRatio, iterTime);
			if (verbosity>4 && pReportWriter!=NULL)
				pReportWriter->IterationReport(vec,i);

		} else break;
		if (MinRel>0.0 && MinRel>Rel) break;
		if (MaxIter>0 && i==MaxIter-1)
			break;
	}
	IterCount+=i;
        if(verbosity>0)
          PrintAvgAvoidance(distanceCalculationsRatioSum/IterCount);
	return BestFit;
}

double KMeansAlgorithm::RunKMeansWithSR(Array<OPTFLOAT> &vec, bool print,double MaxTime)
{
	ASSERT(vec.GetSize()==ncols*nclusters);
	//double BestFit=std::numeric_limits<double>::max();
	double Fit=0.0;
	tms tms_end,tms_start;
	times(&tms_start);
	InitDataStructures(vec);
	for(int i=0;;i++) {
		Fit=ComputeMSEAndCorrect(vec);

		times(&tms_end);
		double rtime=((double)(tms_end.tms_utime+tms_end.tms_stime-tms_start.tms_utime-
		tms_start.tms_stime))/(double)sysconf(_SC_CLK_TCK);
		if (rtime>=MaxTime) break;

		if (print)
			printf("i: %d fit: %5.6f time: %5.3f\n ",i,Fit,rtime);
		PeturbVector(vec,rtime,MaxTime);
	}
	return Fit;
}


NaiveKMA::NaiveKMA(CentroidVector &aCV,StdDataset &Data,CentroidRepair *pR) : KMeansAlgorithm(aCV,Data,pR),
					KMeansWithoutMSE(CV,Data,KMeansAlgorithm::IterCount) {
	Center.SetSize(nclusters*ncols);
	Counts.SetSize(nclusters);
	Assignment.SetSize(Data.GetRowCount());
	pOMPReducer=new Log2OpenMPReducer(aCV);
	int nThreads=omp_get_max_threads();
	OMPData.SetSize(nThreads);

#pragma omp parallel
	{
		int i=omp_get_thread_num();
		OMPData[i].row.SetSize(ncols);
	}

}


NaiveKMA::~NaiveKMA() {
	delete pOMPReducer;
}

void NaiveKMA::PrintNumaLocalityInfo() {
	double Loc=NumaLocalityofArray(Assignment);
	kma_printf("NUMA locality of Assignment is %1.2f%%\n",100.0*Loc);
}

void NaiveKMA::InitDataStructures(Array<OPTFLOAT> &vec) {
	int nRows=Data.GetRowCount();

#pragma omp parallel for default(none) firstprivate(nRows)
	for(int i=0;i<nRows;i++)
		Assignment[i]=0;
}

bool NaiveKMA::CorrectWithoutMSE(Array<OPTFLOAT> &vec, long *distanceCount) {
	for(int i=0;i<nclusters*ncols;i++)
		Center[i]=0.0;
	for(int i=0;i<nclusters;i++)
		Counts[i]=0;

	int NThreads=omp_get_max_threads();

	bool bCont=false;
#pragma omp parallel  default(none) shared(vec) reduction(||:bCont)
	{
		bCont=false;
		int ThreadId=omp_get_thread_num();
		ThreadPrivateVector<OPTFLOAT> &myCenter=pOMPReducer->GetThreadCenter(ThreadId);
		ThreadPrivateVector<int> &myCounts=pOMPReducer->GetThreadCounts(ThreadId);
		ThreadPrivateVector<OPTFLOAT> &tpRow=OMPData[ThreadId].row;

		pOMPReducer->ClearThreadData(ThreadId);

		#pragma omp for OMPDYNAMIC
		for(int i=0;i<Data.GetRowCount();i++) {
			CV.ConvertToOptFloat(tpRow,Data.GetRowNew(i));
			OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
			int bestj=-1;
			for(int j=0;j<nclusters;j++) {
				OPTFLOAT ssq=CV.SquaredDistance(j,vec,tpRow);
				if (ssq<minssq) {
					minssq=ssq;
					bestj=j;
				}
			}
			if (Assignment[i]!=bestj) {
				Assignment[i]=bestj;
				bCont=true;
			}
			myCounts[bestj]++;
			CV.AddRow(bestj,myCenter,tpRow);
		}
	}
	pOMPReducer->ReduceToArrays(Center,Counts);
	ReduceMPIData(Center,Counts,bCont);

	for(int i=0;i<nclusters;i++) {
		if (Counts[i]>0) {
			double f=1.0/Counts[i];
			for(int j=ncols*i;j<ncols*(i+1);j++)
				vec[j]=Center[j]*f;
		} else {
			pRepair->RepairVec(vec,i);
		}
	}
	if (distanceCount!=NULL)
		(*distanceCount)+=((long)Data.GetTotalRowCount()*nclusters);
	return bCont;

}


double NaiveKMA::ComputeMSEAndCorrect(Array<OPTFLOAT> &vec, long *distanceCount)
{
	for(int i=0;i<nclusters*ncols;i++)
		Center[i]=0.0;
	for(int i=0;i<nclusters;i++)
		Counts[i]=0;

	int NThreads=omp_get_max_threads();

	EXPFLOAT fit=(EXPFLOAT)0.0;
#pragma omp parallel  shared(vec) reduction(+:fit)
	{
		fit=0.0;
		int ThreadId=omp_get_thread_num();
		ThreadPrivateVector<OPTFLOAT> &myCenter=pOMPReducer->GetThreadCenter(ThreadId);
		ThreadPrivateVector<int> &myCounts=pOMPReducer->GetThreadCounts(ThreadId);
		ThreadPrivateVector<OPTFLOAT> &tpRow=OMPData[ThreadId].row;

		pOMPReducer->ClearThreadData(ThreadId);
#pragma omp for OMPDYNAMIC
		for(int i=0;i<Data.GetRowCount();i++) {
			CV.ConvertToOptFloat(tpRow,Data.GetRowNew(i));
			OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
			int bestj=-1;
			for(int j=0;j<nclusters;j++) {
				OPTFLOAT ssq=CV.SquaredDistance(j,vec,tpRow);
				if (ssq<minssq) {
					minssq=ssq;
					bestj=j;
				}
			}
			fit+=minssq;
			myCounts[bestj]++;
			CV.AddRow(bestj,myCenter,tpRow);
		}
	}
	pOMPReducer->ReduceToArrays(Center,Counts);
	ReduceMPIData(Center,Counts,fit);

	for(int i=0;i<nclusters;i++) {
		if (Counts[i]>0) {
			double f=1.0/Counts[i];
			for(int j=ncols*i;j<ncols*(i+1);j++)
				vec[j]=Center[j]*f;
		} else {
			pRepair->RepairVec(vec,i);
		}
	}
	if (distanceCount!=NULL)
		(*distanceCount)+=((long)Data.GetTotalRowCount()*nclusters);
	return (double)fit/(double)Data.GetTotalRowCount();
}
