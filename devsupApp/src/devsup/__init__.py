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

    with tempfile.NamedTemporaryFile() as F:
        F.write("""
device(longin, INST_IO, pydevsupComIn, "Python Device")
device(longout, INST_IO, pydevsupComOut, "Python Device")

device(ai, INST_IO, pydevsupComIn, "Python Device")
device(ao, INST_IO, pydevsupComOut, "Python Device")

device(stringin, INST_IO, pydevsupComIn, "Python Device")
device(stringout, INST_IO, pydevsupComOut, "Python Device")

device(bi, INST_IO, pydevsupComIn, "Python Device")
device(bo, INST_IO, pydevsupComOut, "Python Device")

device(mbbi, INST_IO, pydevsupComIn, "Python Device")
device(mbbo, INST_IO, pydevsupComOut, "Python Device")

device(mbbiDirect, INST_IO, pydevsupComIn, "Python Device")
device(mbboDirect, INST_IO, pydevsupComOut, "Python Device")

device(waveform, INST_IO, pydevsupComIn, "Python Device")
device(aai, INST_IO, pydevsupComIn, "Python Device")
device(aao, INST_IO, pydevsupComOut, "Python Device")
""".encode('ascii'))
        F.flush()
        _dbapi.dbReadDatabase(F.name)
    _dbapi._dbd_setup()

def _fini(iocMain=False):
    if iocMain:
        _dbapi.initHookAnnounce(9999) # our magic/fake AtExit hook
    _dbapi._dbd_cleanup()
