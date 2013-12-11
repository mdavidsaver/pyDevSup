Runtime Environment
===================

The pyDevSup module initializes the interpreter during the registration
phase of IOC startup with the *pySetupReg* registrar function. ::

  #!../../bin/linux-x86_64/softIocPy2.7
  # Interpreter not started
  dbLoadDatabase("../../dbd/softIocPy.dbd",0,0)
  softIocPy_registerRecordDeviceDriver(pdbbase)
  # Interpreter started

An *epicsAtExit* cleanup handler is also registered at this time.

Several IOC Shell functions are also registered.
*py()* executes a line of python code.
*pyfile()* reads and interprets a python script. ::

  py("import mymod;)
  py("mymod.makeDev('one')")
  py("mymod.makeDev('two')")
  
  pyfile("myinit.py")

PYTHONPATH
----------

The interpreter search path is automatically prefixed with the following additional paths:

* ``$(EPICS_BASE)/python<PY_VER>/$(EPICS_ARCH)``
* ``$(PYDEV_BASE)/python<PY_VER>/$(EPICS_ARCH)``
* ``$(TOP)/python<PY_VER>/$(EPICS_ARCH)``

The environment variables ``EPICS_BASE``, ``PYDEV_BASE``, ``TOP``, and ``EPICS_ARCH``
will be used if set.  Compile time defaults are selected for
``EPICS_BASE``, ``PYDEV_BASE``, ``EPICS_ARCH``.
If ``TOP`` is not set then the this entry is omitted.

The default for ``PYDEV_BASE`` is the ``$(INSTALL_LOCATION)`` given when the
pyDevSup module was built.

Build Environment
=================

When building IOCs or installing files, include ``PYDEVSUP`` in your *configure/RELEASE*
file. ::

  PYDEVSUP=/dir/where/pyDevSup/is/installed
  EPICS_BASE=/....

The default or preferred Python version can be specificed in *configure/CONFIG_SITE* ::

  PY_VER ?= 2.7

The following should be added to individual EPICS Makefiles. ::

  TOP=../..
  include $(TOP)/configure/CONFIG
  include $(PYDEVSUP)/configure/CONFIG_PY
  ...
  include $(TOP)/configure/RULES
  include $(PYDEVSUP)/configure/RULES_PY

This will add or ammend several make variables.  The ``USR_*FLAGS`` variables
may be extended with approprate flags for building python modules.  The ``PY_VER``
variable is defined with the Python version number found in install directories (eg "2.7").
The ``PY_LD_VER`` variable is defined with the python library version number (eg "3.2mu"),
which may be the same as ``PY_VER``.

Include pyDevSup in your IOC
----------------------------

While the *softIocPyX.Y* executable(s) built as part of this module
will be sufficient for many uses, it is also possible to
include the pyDevSup modules in an IOC along with other drivers.
This can be done in the usual way. ::

  PROD_IOC = myioc
  DBD += myioc.dbd
  ...
  myioc_DBD += pyDevSup.dbd
  ...
  myioc_LIBS += pyDevSup$(PY_LD_VER)

Installing .py files
--------------------

Additional .py files can be installed as follows. ::

  PY += mymod/__init__.py
  PY += mymod/file1.py

Building extensions
-------------------

For convienance, additional Python extensions can be build by the EPICS
build system.  In this example the extension name is "_myextname" and
the resulting library is expected to provide the an initialization function
named "init_myextname". ::

  LOADABLE_LIBRARY_HOST = _myextname

  _myextname_SRCS += somefile.c

In somefile.c ::

  PyMODINIT_FUNC init_myextname(void)

Installing for several Python versions
------------------------------------

The recipe for building and installing the pyDevSup module
for several python version side by side is ::

  make PY_VER=2.6
  make clean
  make PY_VER=2.7
  make clean
  make PY_VER=3.2
  make clean

The ``PYTHON`` make variable can be specified if the interpreter executable
has a name other than ``python$(PY_VER)``.
