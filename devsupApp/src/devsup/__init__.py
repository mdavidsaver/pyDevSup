import os
import sys
import atexit
import tempfile

if sys.platform == 'win32':
    # See https://stackoverflow.com/questions/72858093/how-to-specify-pyd-dll-dependencies-search-paths-at-runtime-with-python
    # This is required for use of e.g. nose testing, but
    # not when running as an IOC, since the IOC will already have loaded EPICS base DLLs.
    xepics_base = None
    xepics_base = os.getenv('XEPICS_BASE')
    print("xepics_base = " + str(xepics_base))
    epics_host_arch = None
    epics_host_arch = os.getenv('EPICS_HOST_ARCH')
    print("epics_host_arch = " + str(epics_host_arch))
    if xepics_base is not None and epics_host_arch is not None:
        xepics_base = xepics_base.strip()
        xepics_base = xepics_base.replace("/",'\\')
        dll_path = xepics_base + "\\bin\\" + epics_host_arch
        print("dll_path = " + dll_path)
        os.add_dll_directory(dll_path)

from . import _dbapi

from ._dbapi import (EPICS_VERSION_STRING,
                     EPICS_DEV_SNAPSHOT,
                     EPICS_SITE_VERSION,
                     EPICS_VERSION,
                     EPICS_REVISION,
                     EPICS_MODIFICATION,
                     EPICS_PATCH_LEVEL,
                     XEPICS_ARCH,
                     XPYDEV_BASE,
                     XEPICS_BASE,
                     epicsver,
                     pydevver,
                     NO_ALARM,
                     MINOR_ALARM,
                     MAJOR_ALARM,
                     READ_ALARM,
                     WRITE_ALARM,
                     HIHI_ALARM,
                     HIGH_ALARM,
                     LOLO_ALARM,
                     LOW_ALARM,
                     STATE_ALARM,
                     COS_ALARM,
                     COMM_ALARM,
                     TIMEOUT_ALARM,
                     HW_LIMIT_ALARM,
                     CALC_ALARM,
                     SCAN_ALARM,
                     LINK_ALARM,
                     SOFT_ALARM,
                     BAD_SUB_ALARM,
                     UDF_ALARM,
                     DISABLE_ALARM,
                     SIMM_ALARM,
                     READ_ACCESS_ALARM,
                     WRITE_ACCESS_ALARM,
                     INVALID_ALARM,
                    )

__all__ = []


def _init(iocMain=False):
    if not iocMain:
        # we haven't read/register base.dbd
        _dbapi.dbReadDatabase(os.path.join(XEPICS_BASE, "dbd", "base.dbd"),
                              path=os.path.join(XEPICS_BASE, "dbd"))
        _dbapi._dbd_rrd_base()
        
    dirname = os.path.dirname(__file__)
    dbd_name = dirname + "/_dbapi.dbd"
    _dbapi.dbReadDatabase(dbd_name)
    epics_version_int = (EPICS_VERSION, EPICS_REVISION, EPICS_MODIFICATION, EPICS_PATCH_LEVEL)
    if epics_version_int >= (3, 15, 0, 2):
        dbd_name = dirname + "/_lsilso.dbd"
        _dbapi.dbReadDatabase(dbd_name)
    if epics_version_int >= (3, 16, 1, 0):
        dbd_name = dirname + "/_int64.dbd"
        _dbapi.dbReadDatabase(dbd_name)
    _dbapi._dbd_setup()


def _fini(iocMain=False):
    if iocMain:
        _dbapi.initHookAnnounce(9999) # our magic/fake AtExit hook
    _dbapi._dbd_cleanup()
