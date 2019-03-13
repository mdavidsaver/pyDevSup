TOP=../..

include $(TOP)/configure/CONFIG
include $(TOP)/configure/CONFIG_PY
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

INSTALL_SHRLIB = $(PY_INSTALL_DIR)/devsup

LOADABLE_LIBRARY_HOST += _dbapi

TARGETS += $(COMMON_DIR)/pyDevSupCommon.dbd
DBDDEPENDS_FILES += pyDevSupCommon.dbd$(DEP)

pyDevSupCommon_DBD += base.dbd

dbapi_CPPFLAGS += -DXEPICS_ARCH=\"$(T_A)\"
dbapi_CPPFLAGS += -DXPYDEV_BASE=\"$(abspath $(INSTALL_LOCATION))\"
dbapi_CPPFLAGS += -DXEPICS_BASE=\"$(EPICS_BASE)\"
dbapi_CPPFLAGS += -DPYDIR=\"python$(PY_VER)\"

_dbapi_SRCS += dbapi.c
_dbapi_SRCS += dbrec.c
_dbapi_SRCS += dbfield.c
_dbapi_SRCS += dbdset.c
_dbapi_SRCS += utest.c

_dbapi_SRCS += pyDevSupCommon_registerRecordDeviceDriver.cpp

_dbapi_LIBS += $(EPICS_BASE_IOC_LIBS)

PY += devsup/__init__.py
PY += devsup/db.py
PY += devsup/dset.py
PY += devsup/hooks.py
PY += devsup/interfaces.py
PY += devsup/util.py
PY += devsup/disect.py
PY += devsup/ptable.py

PY += devsup/test/__init__.py
PY += devsup/test/util.py
PY += devsup/test/test_db.py

#===========================

include $(TOP)/configure/RULES
include $(TOP)/configure/RULES_PY
#----------------------------------------
#  ADD RULES AFTER THIS LINE

pyconfig:
	@echo "Python Configuration for interpreter: $(PYTHON)"
	@echo "Version: $(PY_VER)"
	@echo "Found numpy: $(HAVE_NUMPY)"
	@echo "Includes: $(PY_INCDIRS)"
	@echo "Library path: $(PY_LIBDIRS)"
	@echo "USR_CPPFLAGS: $(USR_CPPFLAGS)"
	@echo "USR_LDFLAGS: $(USR_LDFLAGS)"

ifneq (,$(T_A))
nose:
	PYTHONPATH="${PYTHONPATH}:$(abspath $(TOP))/python$(PY_LD_VER)/$(EPICS_HOST_ARCH)" $(PYTHON) -m nose -P devsup $(NOSEFLAGS)

# bounce back down to the sphinx generated Makefile
# aren't Makefiles fun...
sphinx:
	PYTHONPATH="${PYTHONPATH}:$(abspath $(TOP))/python$(PY_LD_VER)/$(EPICS_HOST_ARCH)" $(MAKE) -C $(TOP)/documentation html

sh:
	echo "export PYTHONPATH=\$${PYTHONPATH}:$(abspath $(TOP))/python$(PY_LD_VER)/$(EPICS_HOST_ARCH)" > $(OUTPUT)

ipython:
	PYTHONPATH="${PYTHONPATH}:$(abspath $(TOP))/python$(PY_LD_VER)/$(EPICS_HOST_ARCH)" $(PYTHON) -c "import sys; sys.argv[0] = '$(PYTHON)'; from IPython.terminal.ipapp import launch_new_instance; launch_new_instance()"

endif
