try:
    import _dbapi
except ImportError:
    import devsup._nullapi as _dbapi

__all__ = ['verinfo']

verinfo = _dbapi.verinfo
