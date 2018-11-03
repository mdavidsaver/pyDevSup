#Makefile at top of application tree
TOP = .
include $(TOP)/configure/CONFIG
DIRS := $(DIRS) $(filter-out $(DIRS), configure)
DIRS := $(DIRS) $(filter-out $(DIRS), $(wildcard *App))
DIRS := $(DIRS) $(filter-out $(DIRS), $(wildcard iocBoot))

define DIR_template
 $(1)_DEPEND_DIRS = configure
endef
$(foreach dir, $(filter-out configure,$(DIRS)),$(eval $(call DIR_template,$(dir))))

pyIocApp_DEPEND_DIRS += devsupApp

iocBoot_DEPEND_DIRS += $(filter %App,$(DIRS))

include $(TOP)/configure/RULES_TOP

UNINSTALL_DIRS += $(wildcard $(INSTALL_LOCATION)/python*)

# jump to a sub-directory where CONFIG_PY has been included
# can't include CONFIG_PY here as it may not exist yet
nose sphinx sh ipython: all
	$(MAKE) -C devsupApp/src/O.$(EPICS_HOST_ARCH) $@ PYTHON=$(PYTHON)

sphinx-clean:
	$(MAKE) -C documentation clean PYTHON=$(PYTHON)

sphinx-commit: sphinx
	touch documentation/_build/html/.nojekyll
	./commit-gh.sh documentation/_build/html

.PHONY: nose sphinx sphinx-commit sphinx-clean
