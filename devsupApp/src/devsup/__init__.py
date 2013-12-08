try:
    import _dbapi
    HAVE_DBAPI = True
except ImportError:
    import devsup._nullapi as _dbapi
    HAVE_DBAPI = False

try:
    from _dbconstants import *
except ImportError:
    EPICS_VERSION_STRING = "EPICS 0.0.0.0-0"
    EPICS_DEV_SNAPSHOT = ""
    EPICS_SITE_VERSION = "0"
    EPICS_VERSION = 0
    EPICS_REVISION = 0
    EPICS_MODIFICATION = 0
    EPICS_PATCH_LEVEL = 0

    XEPICS_ARCH = "nullos-nullarch"
    XPYDEV_BASE = "invaliddir"
    XEPICS_BASE = "invaliddir"

    epicsver = (0,0,0,0,"0","")
    pydevver = (0,0)

__all__ = []
