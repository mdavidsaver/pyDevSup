TOP=../..

include $(TOP)/configure/CONFIG
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

ifdef T_A
include $(TOP)/configure/O.$(T_A)/CONFIG_PY
ifneq ($(PY_OK),YES)
$(error Unable to get python configuration)
endif
endif

USR_CPPFLAGS += $(PY_INCDIRS:%=-I%)
USR_LDFLAGS += $(PY_LIBDIRS:%=-L%)

ifeq ($(HAVE_NUMPY),YES)
USR_CPPFLAGS += -DHAVE_NUMPY
endif

#=============================
# Build the IOC application

PROD_IOC = devsup
# devsup.dbd will be created and installed
DBD += devsup.dbd

# devsup.dbd will be made up from these files:
devsup_DBD += base.dbd
devsup_DBD += pyDevSup.dbd

devsup_SYS_LIBS += python$(PY_VER)

# devsup_registerRecordDeviceDriver.cpp derives from devsup.dbd
devsup_SRCS += devsup_registerRecordDeviceDriver.cpp

# Build the main IOC entry point on workstation OSs.
devsup_SRCS_DEFAULT += devsupMain.cpp
devsup_SRCS_vxWorks += -nil-

devsup_SRCS += setup.c
devsup_SRCS += dbrec.c
devsup_SRCS += dbfield.c
devsup_SRCS += dbdset.c

# Add support from base/src/vxWorks if needed
#devsup_OBJS_vxWorks += $(EPICS_BASE_BIN)/vxComLibrary

# Finally link to the EPICS Base libraries
devsup_LIBS += $(EPICS_BASE_IOC_LIBS)


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
