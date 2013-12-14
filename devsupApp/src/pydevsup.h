#ifndef PYDEVSUP_H
#define PYDEVSUP_H

#include <epicsThread.h>

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#define PyString_FromString PyUnicode_FromString
#define MODINIT_RET(VAL) return (VAL)

#else
#define MODINIT_RET(VAL) return

#endif
void pyDBD_cleanup(void);

int pyField_prepare(PyObject *module);
void pyField_cleanup(void);

int pyRecord_prepare(PyObject *module);

int isPyRecord(dbCommon *);
int canIOScanRecord(dbCommon *);

extern epicsThreadPrivateId pyDevReasonID;

#endif // PYDEVSUP_H
