TOP=../..

include $(TOP)/configure/CONFIG
PYMODULE = NO
include $(TOP)/configure/CONFIG_PY
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

#=============================
# Build the IOC application

LIBRARY = pyDevSup$(PY_VER)

SHRLIB_VERSION = 0

DBD += pyDevSup.dbd

pyDevSup$(PY_VER)_SYS_LIBS += python$(PY_VER)

pyDevSup$(PY_VER)_SRCS += setup.c
pyDevSup$(PY_VER)_SRCS += dbrec.c
pyDevSup$(PY_VER)_SRCS += dbfield.c
pyDevSup$(PY_VER)_SRCS += dbdset.c


PROD_IOC = softIocPy
# softIocPy.dbd will be created and installed
DBD += softIocPy.dbd

# softIocPy.dbd will be made up from these files:
softIocPy_DBD += base.dbd
softIocPy_DBD += pyDevSup.dbd

softIocPy_LIBS += pyDevSup$(PY_VER)

softIocPy_SYS_LIBS += python$(PY_VER)

# softIocPy_registerRecordDeviceDriver.cpp derives from softIocPy.dbd
softIocPy_SRCS += softIocPy_registerRecordDeviceDriver.cpp

# Build the main IOC entry point on workstation OSs.
softIocPy_SRCS_DEFAULT += devsupMain.cpp
softIocPy_SRCS_vxWorks += -nil-

# Add support from base/src/vxWorks if needed
#softIocPy_OBJS_vxWorks += $(EPICS_BASE_BIN)/vxComLibrary

# Finally link to the EPICS Base libraries
softIocPy_LIBS += $(EPICS_BASE_IOC_LIBS)


PY += devsup/__init__.py
PY += devsup/_nullapi.py
PY += devsup/db.py
PY += devsup/hooks.py
PY += devsup/interfaces.py
PY += devsup/util.py

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
