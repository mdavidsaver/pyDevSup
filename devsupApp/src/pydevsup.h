#ifndef PYDEVSUP_H
#define PYDEVSUP_H

#include <epicsVersion.h>
#include <epicsThread.h>
#include <initHooks.h>

#include <Python.h>

#ifndef VERSION_INT
#  define VERSION_INT(V,R,M,P) ( ((V)<<24) | ((R)<<16) | ((M)<<8) | (P))
#endif
#ifndef EPICS_VERSION_INT
#  define EPICS_VERSION_INT VERSION_INT(EPICS_VERSION, EPICS_REVISION, EPICS_MODIFICATION, EPICS_PATCH_LEVEL)
#endif

// handle -fvisibility=default
#if __GNUC__ >= 4
#  undef PyMODINIT_FUNC
#  if defined(__cplusplus)
#    if PY_MAJOR_VERSION < 3
#      define PyMODINIT_FUNC extern "C" __attribute__ ((visibility("default"))) void
#    else
#      define PyMODINIT_FUNC extern "C" __attribute__ ((visibility("default"))) PyObject*
#    endif
#  else
#    if PY_MAJOR_VERSION < 3
#      define PyMODINIT_FUNC __attribute__ ((visibility("default"))) void
#    else
#      define PyMODINIT_FUNC __attribute__ ((visibility("default"))) PyObject*
#    endif
#  endif
#endif

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#define PyString_FromString PyUnicode_FromString
#define MODINIT_RET(VAL) return (VAL)

#else
#define MODINIT_RET(VAL) return

#endif

struct dbCommon;

PyObject* pyDBD_setup(PyObject *unused);
PyObject* pyDBD_cleanup(PyObject *unused);

int pyUTest_prepare(PyObject *module);

int pyField_prepare(PyObject *module);

int pyRecord_prepare(PyObject *module);

int isPyRecord(struct dbCommon *);
int canIOScanRecord(struct dbCommon *);

extern epicsThreadPrivateId pyDevReasonID;

#endif // PYDEVSUP_H
