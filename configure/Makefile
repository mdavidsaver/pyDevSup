TOP=..

include $(TOP)/configure/CONFIG

TARGETS = $(CONFIG_TARGETS)
CONFIGS += $(subst ../,,$(wildcard $(CONFIG_INSTALLS)))

TARGETS += CONFIG_PY RULES_PY

include $(TOP)/configure/RULES

CONFIG_PY: $(TOP)/makehelper.py
	$(PYTHON) $< $@

RULES_PY: ../RULES_PY
	install -m644 $< $@
