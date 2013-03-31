try:
    import _dbapi
except ImportError:
    import _nullapi as _dbapi

__all__ = ['verinfo']

verinfo = _dbapi.verinfo
