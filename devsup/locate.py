"""
Locate an EPICS installation
"""


import os, os.path, platform

__all__ = [
  'base'
  'hostarch',
  'binpath',
  'dbpath',
  'dbdpath',
  'cpppath',
]

# Existance of these files is required
_files = ['include/epicsVersion.h', 'dbd/base.dbd']

_possible = [
  '/usr/local/epics/base',
  '/usr/local/epics/base/current',
  '/usr/local/lib/epics',
  '/opt/epics',
  '/opt/epics/base',
  '/opt/epics/base/current',
  '/usr/lib/epics',
]

try:
  envbase = os.environ['EPICS_BASE']
  if os.path.isdir(envbase):
    _possible.insert(-1, envbase)
  else:
    import warnings
    warnings.warn("'%s' does not name a directory"%envbase, RuntimeWarning)
  del envbase
except KeyError:
  pass

def _findbase(poss):
  for p in poss:
    if not os.path.isdir(p):
      continue
    for f in _files:
      fp = os.path.join(p,f)
      if not os.path.isfile(fp):
        break
    return p
  raise RuntimeError("Failed to locate EPICS Base")

base = _findbase(_possible)
del _possible
del _findbase

try:
  _dbg = bool(int(os.environ.get('EPICS_DEBUG', '0')))
except ValueError:
  _dbg = False


cpppath = [os.path.join(base,'include')]

def _findarch():
  _plat = platform.system()
  if _plat=='Windows':
    cpppath.append(os.path.join(base, 'os', 'WIN32'))
    raise RuntimeError("Windows support not complete")

  elif _plat=='Linux':
    cpppath.append(os.path.join(base, 'include', 'os', _plat))
    archs = {'32bit':'x86', '64bit':'x86_64'}
    arch = archs[platform.architecture()[0]]

    return 'linux-%s%s' % (arch, '-debug' if _dbg else '')
    
  else:
    raise RuntimeError('Unsupported platform %s'%_plat)

hostarch = _findarch()
del _findarch

ldpath = [os.path.join(base,'lib',hostarch)]
binpath = [os.path.join(base,'bin',hostarch)]
dbpath = [os.path.join(base,'db',hostarch)]
dbdpath = [os.path.join(base,'dbd',hostarch)]
