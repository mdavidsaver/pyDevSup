#!/usr/bin/env python
"""
Set some python derived Makefile variables.

Emits something like the following

PY_OK := YES  # indicates success of this script
HAVE_NUMPY := YES/NO
PY_VER := 2.6
PY_INCDIRS := /path ...
PY_LIBDIRS := /path ...
"""

from __future__ import print_function

import sys

if len(sys.argv)<2:
    out = sys.stdout
else:
    out = open(sys.argv[1], 'w')

from distutils.sysconfig import get_config_var, get_python_inc

incdirs = [get_python_inc()]
libdirs = [get_config_var('LIBDIR')]

have_np='NO'
try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    incdirs += get_numpy_include_dirs()
    have_np='YES'
except ImportError:
    pass

incdirs = [get_python_inc()]+get_numpy_include_dirs()
libdirs = [get_config_var('LIBDIR')]

print('PY_VER :=',get_config_var('VERSION'), file=out)
print('PY_INCDIRS :=',' '.join(incdirs), file=out)
print('PY_LIBDIRS :=',' '.join(libdirs), file=out)
print('HAVE_NUMPY :=',have_np, file=out)

print('PY_OK := YES', file=out)

out.close()
