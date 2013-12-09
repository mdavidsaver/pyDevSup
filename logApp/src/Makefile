TOP=../..

include $(TOP)/configure/CONFIG
include $(TOP)/configure/CONFIG_PY
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

#=============================
# Build the IOC application

LOADABLE_LIBRARY_HOST = _inotifyy

_inotifyy_SRCS += inotify_wrap.c

PY += inotifyy.py
PY += logwatch.py

#===========================

include $(TOP)/configure/RULES
include $(TOP)/configure/RULES_PY
#----------------------------------------
#  ADD RULES AFTER THIS LINE
