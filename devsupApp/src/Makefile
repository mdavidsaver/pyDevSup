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

_dbapi_SRCS += pyDevSupCommon_registerRecordDeviceDriver.cpp

PY += devsup/__init__.py
PY += devsup/_nullapi.py
PY += devsup/db.py
PY += devsup/dset.py
PY += devsup/hooks.py
PY += devsup/interfaces.py
PY += devsup/util.py
PY += devsup/disect.py
PY += devsup/ptable.py

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
