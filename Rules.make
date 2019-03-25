%.dep : %.cpp
		$(CC) -M $< > $@
%.ps  : %
		a2ps --highlight-level=none -T 4 -f 6 -M a4  $< -o $@

%.o : %.cpp
ifeq ($(VERBOSE),y)
		$(CC) $(COPT) $(INCLUDE) $< -o $@

else
		@echo [CC] $@
		@$(CC) $(COPT) $(INCLUDE) $< -o $@
endif

ifeq (depend,$(wildcard depend))
include depend
endif
