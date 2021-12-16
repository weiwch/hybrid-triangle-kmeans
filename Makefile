include Rules.make
# determine the hardware platform e.g. i686 or ia64 (itanium)
ARCH=$(shell uname -m)
BRANCH=$(shell git rev-parse --abbrev-ref HEAD)


mpikmeans : override CC=mpicxx

ifeq ($(MPE),y)
mpikmeans : override CC=mpecc -mpilog -pthread -mpicc=mpicxx
FIX5=_mpe
endif


mpikmeans : override COPT+=-D_MPI


# select options for chosen compiler

ifeq ($(LONGEXP),y)
EXPFLOAT=-DEXPFLOAT=long\ double
FIX2=_long
endif

ifeq ($(OPENMP),y)
FIX3=_openmp
ifeq ($(DYNAMIC),y)
FIX3a=_dynamic
endif
endif


#ifeq ($(GCC),y)
#gcc options

CC=g++

ifeq ($(OPENMP),y)
OMP=-fopenmp
endif
ifeq ($(SCALASCA),y)
CC=scorep --instrument-filter=scorep.filter g++
mpikmeans : override CC=scorep  --instrument-filter=scorep.filter mpicxx
endif


COMMONOPT=$(OMP) -c $(EXPFLOAT) -std=gnu++11 -DBRANCH=\"$(BRANCH)\"
#architecture spe=ific Options
include Options-$(ARCH).gcc



#endif


ifeq ($(ICC),y)
#icc options

ifeq ($(OPENMP),y)
OMP=-fopenmp
endif


COMMONOPT=-c $(EXPFLOAT) $(OMP) -std=gnu++11 -DBRANCH=\"$(BRANCH)\"
CC=icpc

ifeq ($(SCALASCA),y)
CC=scorep --nocompiler icpc
mpikmeans : override CC=scorep --nocompiler mpicxx
endif

ifeq ($(TAU),y)
CC=tau_cxx.sh
mpikmeans : override CC=tau_cxx.sh
endif

#architecture specific Options
include Options-$(ARCH).icc
endif


ifeq ($(PGI),y)
CC=pgc++
COMMONOPT=-c $(EXPFLOAT)
include Options-$(ARCH).pgi
ifeq ($(OPENMP),y)
CC:=$(CC) -mp
endif
endif


ifeq ($(SUN),y)
CC=CC
COMMONOPT=-c
include Options-$(ARCH).sun
ifeq ($(OPENMP),y)
CC:=$(CC) -xopenmp=parallel
endif
endif

ifeq ($(OPN),y)
CC=openCC
COMMONOPT=-c
include Options-$(ARCH).opn
ifeq ($(OPENMP),y)
CC:=$(CC) -mp
endif
endif


ifeq ($(OPT),y)
FIX=_release
ifeq ($(MPE),y)
COPT=$(RELEASE) -D_MPE
else
COPT=$(RELEASE)
endif
else
COPT=$(DEBUG)
FIX=_debug
endif


COPT:=$(COPT) -DLARGEVECTOR

ifeq ($(NO_S),y)
	COPT:=$(COPT) -DNO_S
endif

ifeq ($(NO_U),y)
	COPT:=$(COPT) -DNO_U
endif

ifeq ($(OPTDOUBLE),y)
COPT:=$(COPT) -DOPTFLOAT=double
endif

ifeq ($(DYNAMIC),y)
COPT:=$(COPT) -DOMPDYNAMIC=schedule\(guided\)
endif



COPT:=$(COPT) $(PROFOPT)
LOPT:=$(LOPT) $(OMP) $(PROFOPT)


UTIL_O = $(addprefix Util/, Array.o Dataset.o Debug.o Rand.o  StdDataset.o Partition.o NumaDataset.o NumaAlloc.o PrecisionTimer.o)
CLUST_O= $(addprefix Clust/,KMeansInitializer.o CentroidVector.o KMAlgorithm.o  \
HamerlyKMA/HamerlyKMA.o HamerlyKMA/HamerlySmallestDistances.o ElkanKMA/ElkanSmallestDistances.o ElkanKMA/ElkanKMA.o AnnulusKMA/AnnulusKMA.o DrakeKMA/DrakeKMA.o KMeansWithoutMSE/KMeansWithoutMSE.o \
PlusPlusInitializer/PlusPlusInitializer.o YinyangKMA.o CentroidVectorPermutation.o CentroidRepair.o \
OpenMPKMAReducer.o SameSizeKMA.o YinyangClusterer.o KMeansOrOrInitializer.o KMeansReportWriter.o) Util/StdOut.o

MPIKMA_O=MPIKmeans/DistributedNumaDataset.o MPIKmeans/MPIForgyInitializer.o MPIKmeans/MPINaiveKMA.o MPIKmeans/MPIKMAReducer.o\
MPIKmeans/MPIRank0StdOut.o MPIKmeans/MPIElkanKMA/MPIElkanKMA.o MPIKmeans/MPIDrakeKMA/MPIDrakeKMA.o MPIKmeans/MPIHamerlyKMA/MPIHamerlyKMA.o MPIKmeans/MPIAnnulusKMA/MPIAnnulusKMA.o \
MPIKmeans/MPIYinyangKMA.o MPIKmeans/MPICentroidRandomRepair.o Clust/YinyangKMA.o Clust/PlusPlusInitializer/PlusPlusInitializer.o \
MPIKmeans/MPIItemDistribution.o MPIKmeans/MPIHamerlyKMA/MPIHamerlySmallestDistances.o \
MPIKmeans/MPIElkanKMA/MPIElkanSmallestDistances.o Clust/YinyangClusterer.o  Clust/KMeansReportWriter.o \
MPIKmeans/MPIKMeansOrOrInitializer.o Clust/KMeansOrOrInitializer.o \
$(addprefix Clust/, KMeansInitializer.o CentroidVector.o KMAlgorithm.o CentroidVectorPermutation.o\
  CentroidRepair.o SameSizeKMA.o KMeansWithoutMSE/KMeansWithoutMSE.o ElkanKMA/ElkanKMA.o ElkanKMA/ElkanSmallestDistances.o HamerlyKMA/HamerlyKMA.o HamerlyKMA/HamerlySmallestDistances.o AnnulusKMA/AnnulusKMA.o DrakeKMA/DrakeKMA.o OpenMPKMAReducer.o) \
Util/StdOut.o







kmeans : $(UTIL_O) $(CLUST_O) Clust/mainkm.o
ifeq ($(VERBOSE),y)	
	$(CC)   $^  $(LOPT) -lnuma -o $@
else
	@echo [CCLD] $@
	@$(CC) $^  -lnuma $(LOPT) -o $@
endif	
	mv kmeans Object/kmeans$(FIX3)$(FIX3a)$(FIX)_$(BRANCH)

kminit : $(UTIL_O) $(CLUST_O) Clust/kminit.o
ifeq ($(VERBOSE),y)	
	$(CC)   $^  $(LOPT) -lnuma -o $@
else
	@echo [CCLD] $@
	@$(CC) $^  -lnuma $(LOPT) -o $@
endif	
	mv kminit Object/kminit$(FIX3)$(FIX3a)$(FIX)_$(BRANCH)

mpikmeans : $(UTIL_O) $(MPIKMA_O) MPIKmeans/main.o
ifeq ($(VERBOSE),y)	
	$(CC)   $^  $(LOPT) -lnuma -o $@
else
	@echo [CCLD] $@
	@$(CC) $^  -lnuma $(LOPT) -o $@
endif	
	mv mpikmeans Object/mpikmeans$(FIX3)$(FIX3a)$(FIX)_$(BRANCH)
	

clean:
	rm -rf `find . -name "*.o"` 
	rm -rf kmeans Object/kmeans*
	rm -rf kminit Object/kminit*
	rm -rf mpikmeans Object/mpikmeans*


install:
	install -d ~/bin
ifneq ("$(wildcard Object/*release*)","")
	install Object/*release* ~/bin
endif

ifneq ("$(wildcard /srv/grid/$(USER)/bin)","")
	install ~/bin/*  /srv/grid/$(USER)/bin
endif


