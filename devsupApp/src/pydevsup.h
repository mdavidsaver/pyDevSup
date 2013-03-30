#ifndef PYDEVSUP_H
#define PYDEVSUP_H

int pyField_prepare(void);
void pyField_setup(PyObject *module);
void PyField_cleanup(void);

int pyRecord_prepare(void);
void pyRecord_setup(PyObject *module);

int isPyRecord(dbCommon *);
int setCausePyRecord(dbCommon *, PyObject *);
int clearCausePyRecord(dbCommon *);

#endif // PYDEVSUP_H
