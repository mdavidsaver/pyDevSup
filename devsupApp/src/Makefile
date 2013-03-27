TOP=../..

include $(TOP)/configure/CONFIG
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

PYTHON = python
# EG 2.7
PY_VER = $(shell $(PYTHON) -c 'from distutils.sysconfig import get_config_var; print get_config_var("VERSION")')
# EG -I/usr/include/python2.7
PY_CFLAGS := -I$(shell $(PYTHON) -c 'from distutils.sysconfig import get_python_inc; print get_python_inc()')
# EG -L/usr/lib
PY_LIBDIR := -L$(shell $(PYTHON) -c 'from distutils.sysconfig import get_config_var; print get_config_var("LIBDIR")')

USR_CPPFLAGS += $(PY_CFLAGS)
USR_LDFLAGS += $(PY_LIBDIR)

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

#===========================

include $(TOP)/configure/RULES
#----------------------------------------
#  ADD RULES AFTER THIS LINE

