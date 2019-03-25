
Util : Debug.o Array.o Dataset.o Rand.o	Standarizer.o
	touch Util

depend :
	$(CC) -M *.cpp  > $@

postscript : Array.h.ps Debug.h.ps Array.cpp.ps Debug.cpp.ps Dataset.h.ps Dataset.cpp.ps \
Rand.h.ps Rand.cpp.ps Standarizer.cpp.ps Standarizer.h.ps

include ../Rules.make
