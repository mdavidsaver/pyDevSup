TOP=..

include $(TOP)/configure/CONFIG

TARGETS = $(CONFIG_TARGETS)

ifdef T_A
CONFIGS = CONFIG_PY RULES_PY os/CONFIG_PY.Common.$(T_A)
endif

include $(TOP)/configure/RULES

os/CONFIG_PY.Common.%: $(TOP)/makehelper.py
	[ -d $(dir $@) ] || $(MKDIR) $(dir $@)
	$(PYTHON) $< $@

ifeq ($(BASE_3_15), YES)
realclean: py_clean
else
realclean:: py_clean
endif

py_clean:
	$(RMDIR) os
