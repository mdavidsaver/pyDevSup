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

realclean:: py_clean
clean:: py_clean

py_clean:
	$(RMDIR) os
