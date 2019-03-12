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

    # Alarm Severity
    (
        NO_ALARM,        # No alarm has been triggered
        MINOR_ALARM,     # Needs attention but is not dangerous (warning)
        MAJOR_ALARM,     # Needs immediate attention by the operator (serious alarm)
        INVALID_ALARM    # Invalid data, e.g. due to device communication failure, overflow, etc.
    ) = range(4)

    # Alarm Status
    (
        NO_ALARM,        # This record is not in alarm
        READ_ALARM,      # An INPUT link failed in the device support
        WRITE_ALARM,     # An OUTPUT link failed in the device support
        HIHI_ALARM,      # An analog value limit alarm
        HIGH_ALARM,      # An analog value limit alarm
        LOLO_ALARM,      # An analog value limit alarm
        LOW_ALARM,       # An analog value limit alarm
        STATE_ALARM,     # A digital value state alarm
        COS_ALARM,       # A digital value change of state alarm
        COMM_ALARM,      # A device support alarm that indicates the device is not communicating
        TIMEOUT_ALARM,   # A device sup alarm that indicates the asynchronous device timed out
        HW_LIMIT_ALARM,  # A device sup alarm that indicates a hardware limit alarm
        CALC_ALARM,      # A record support alarm for calculation records indicating a bad calulation
        SCAN_ALARM,      # An invalid SCAN field is entered
        LINK_ALARM,      # Soft device support for a link failed
        SOFT_ALARM,
        BAD_SUB_ALARM,
        UDF_ALARM,
        DISABLE_ALARM,
        SIMM_ALARM,
        READ_ACCESS_ALARM,
        WRITE_ACCESS_ALARM
    ) = range(22)

__all__ = []
