TOP=../..

include $(TOP)/configure/CONFIG
PYMODULE = NO
include $(TOP)/configure/CONFIG_PY
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

#=============================
# Build the IOC application

LIBRARY = pyDevSup$(PY_LD_VER)

SHRLIB_VERSION = 0

DBD += pyDevSup.dbd

pyDevSup$(PY_LD_VER)_SYS_LIBS += python$(PY_LD_VER)

setup_CPPFLAGS += -DXEPICS_ARCH=\"$(T_A)\"
setup_CPPFLAGS += -DXPYDEV_BASE=\"$(abspath $(INSTALL_LOCATION))\"
setup_CPPFLAGS += -DXEPICS_BASE=\"$(EPICS_BASE)\"
setup_CPPFLAGS += -DPYDIR=\"python$(PY_VER)\"

pyDevSup$(PY_LD_VER)_SRCS += setup.c
pyDevSup$(PY_LD_VER)_SRCS += dbrec.c
pyDevSup$(PY_LD_VER)_SRCS += dbfield.c
pyDevSup$(PY_LD_VER)_SRCS += dbdset.c


PROD_IOC = softIocPy$(PY_VER)
PRODNAME = $(addsuffix $(EXE),$(PROD))

# softIocPy.dbd will be created and installed
DBD += softIocPy.dbd

# softIocPy.dbd will be made up from these files:
softIocPy_DBD += base.dbd
softIocPy_DBD += pyDevSup.dbd

softIocPy$(PY_VER)_LIBS += pyDevSup$(PY_LD_VER)

# softIocPy_registerRecordDeviceDriver.cpp derives from softIocPy.dbd
softIocPy$(PY_VER)_SRCS += softIocPy_registerRecordDeviceDriver.cpp

# Build the main IOC entry point on workstation OSs.
softIocPy$(PY_VER)_SRCS_DEFAULT += devsupMain.cpp
softIocPy$(PY_VER)_SRCS_vxWorks += -nil-

# Add support from base/src/vxWorks if needed
#softIocPy$(PY_VER)_OBJS_vxWorks += $(EPICS_BASE_BIN)/vxComLibrary

# Finally link to the EPICS Base libraries
softIocPy$(PY_VER)_LIBS += $(EPICS_BASE_IOC_LIBS)


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
