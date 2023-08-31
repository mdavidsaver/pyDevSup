import os
import atexit
import tempfile

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
    print("opening database " + dbd_name)
    _dbapi.dbReadDatabase(dbd_name)
    # e.g. 3.15.0.2 -> 30000 + 1500 + 0 + 2 = 31502
    epics_version_int = EPICS_VERSION * 10000 + EPICS_REVISION * 100 + EPICS_MODIFICATION * 10 + EPICS_PATCH_LEVEL
    print("epics_version_int = " + str(epics_version_int))
    if epics_version_int >= 31502:
        dbd_name = dirname + "/_lsilso.dbd"
        print("opening database " + dbd_name)
        _dbapi.dbReadDatabase(dbd_name)
    if epics_version_int >= 31610:
        dbd_name = dirname + "/_int64.dbd"
        print("opening database " + dbd_name)
        _dbapi.dbReadDatabase(dbd_name)
    _dbapi._dbd_setup()


def _fini(iocMain=False):
    if iocMain:
        _dbapi.initHookAnnounce(9999) # our magic/fake AtExit hook
    _dbapi._dbd_cleanup()
