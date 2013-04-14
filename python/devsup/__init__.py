try:
    import _dbapi
except ImportError:
    import devsup._nullapi as _dbapi

try:
    from _dbconstants import *
except ImportError:
    pass

__all__ = ['verinfo']

verinfo = _dbapi.verinfo
