#!/bin/bash





make clean
srun -p test make -j 24 kmeans OPT=y  LONGEXP=y OPTDOUBLE=y LARGEVECTOR=y VERBOSE=y ICC=y
make install


make clean
srun -p test make -j 24 kmeans OPT=y  LONGEXP=y OPTDOUBLE=y LARGEVECTOR=y OPENMP=y ICC=y
make install
make clean

srun -p test make -j 24 kmeans OPT=y  LONGEXP=y OPTDOUBLE=y LARGEVECTOR=y OPENMP=y DYNAMIC=y ICC=y
make install
make clean

srun -p test make -j 24 mpikmeans OPT=y  LONGEXP=y OPTDOUBLE=y LARGEVECTOR=y OPENMP=y ICC=y
make install
make clean

srun -p test make -j 24 mpikmeans OPT=y  LONGEXP=y OPTDOUBLE=y LARGEVECTOR=y ICC=y
make install
make clean

srun -p test make -j 24 mpikmeans OPT=y  LONGEXP=y OPTDOUBLE=y LARGEVECTOR=y OPENMP=y DYNAMIC=y ICC=y
make install
make clean

srun -p test make -j 24 kminit OPT=y  LONGEXP=y OPTDOUBLE=y LARGEVECTOR=y OPENMP=y ICC=y
make install
make clean
