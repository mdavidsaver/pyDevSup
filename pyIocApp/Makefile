TOP=..

include $(TOP)/configure/CONFIG
PYMODULE = NO
include $(TOP)/configure/CONFIG_PY
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

#=============================
# Build the IOC application

PROD_IOC = softIocPy$(PY_VER)
PRODNAME = $(addsuffix $(EXE),$(PROD))

# softIocPy.dbd will be created and installed
DBD += softIocPy.dbd

# softIocPy.dbd will be made up from these files:
softIocPy_DBD += base.dbd
softIocPy_DBD += pyDevSup.dbd

softIocPy_DBD += system.dbd

softIocPy$(PY_VER)_LIBS += pyDevSup$(PY_LD_VER)

# softIocPy_registerRecordDeviceDriver.cpp derives from softIocPy.dbd
softIocPy$(PY_VER)_SRCS += softIocPy_registerRecordDeviceDriver.cpp

# Build the main IOC entry point on workstation OSs.
softIocPy$(PY_VER)_SRCS_DEFAULT += devsupMain.cpp

devsupMain_CPPFLAGS += -DXPYDEV_BASE=\"$(abspath $(INSTALL_LOCATION))\"

ifneq ($(DEVIOCSTATS),)
softIocPy_DBD += devIocStats.dbd
softIocPy$(PY_VER)_LIBS += devIocStats
endif

ifneq ($(AUTOSAVE),)
softIocPy_DBD += asSupport.dbd
softIocPy$(PY_VER)_LIBS += autosave
endif

ifneq ($(CAPUTLOG),)
softIocPy_DBD += caPutLog.dbd
softIocPy$(PY_VER)_LIBS += caPutLog
endif

# Finally link to the EPICS Base libraries
softIocPy$(PY_VER)_LIBS += $(EPICS_BASE_IOC_LIBS)

#===========================

include $(TOP)/configure/RULES
include $(TOP)/configure/RULES_PY
#----------------------------------------
#  ADD RULES AFTER THIS LINE
