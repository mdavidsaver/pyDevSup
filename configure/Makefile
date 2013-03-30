TOP=..

include $(TOP)/configure/CONFIG

TARGETS = $(CONFIG_TARGETS)
CONFIGS += $(subst ../,,$(wildcard $(CONFIG_INSTALLS)))

TARGETS += CONFIG_PY

include $(TOP)/configure/RULES

CONFIG_PY: $(TOP)/makehelper.py
	$(PYTHON) $< $@
