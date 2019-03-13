#ifndef PYDEVSUP_H
#define PYDEVSUP_H

#include <epicsThread.h>
#include <initHooks.h>

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
