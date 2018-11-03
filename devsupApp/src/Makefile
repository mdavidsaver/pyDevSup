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
pyDevSup$(PY_LD_VER)_SRCS += dbbase.c
pyDevSup$(PY_LD_VER)_SRCS += dbrec.c
pyDevSup$(PY_LD_VER)_SRCS += dbfield.c
pyDevSup$(PY_LD_VER)_SRCS += dbdset.c

pyDevSup$(PY_LD_VER)_LIBS += $(EPICS_BASE_IOC_LIBS)

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
