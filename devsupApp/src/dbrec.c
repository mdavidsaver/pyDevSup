
/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

#include <epicsVersion.h>
#include <dbCommon.h>
#include <dbAccess.h>
#include <dbStaticLib.h>
#include <recSup.h>
#include <dbScan.h>

#include "pydevsup.h"

typedef struct {
    PyObject_HEAD

    DBENTRY entry;
} pyRecord;

static void pyRecord_dealloc(pyRecord *self)
{
    dbFinishEntry(&self->entry);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject* pyRecord_new(PyTypeObject *type, PyObject *args, PyObject *kws)
{
    pyRecord *self;

    self = (pyRecord*)type->tp_alloc(type, 0);
    if(self) {
        dbInitEntry(pdbbase, &self->entry);
    }
    return (PyObject*)self;
}

static int pyRecord_Init(pyRecord *self, PyObject *args, PyObject *kws)
{
    const char *recname;
    if(!PyArg_ParseTuple(args, "s", &recname))
        return -1;

    if(dbFindRecord(&self->entry, recname)){
        PyErr_SetString(PyExc_ValueError, "No record by this name");
        return -1;
    }
    return 0;
}

static int pyRecord_compare(pyRecord *A, pyRecord *B)
{
    dbCommon *a=A->entry.precnode->precord,
             *b=B->entry.precnode->precord;

    if(a==b)
        return 0;
    return strcmp(a->name, b->name);
}

static PyObject* pyRecord_name(pyRecord *self)
{
    dbCommon *prec=self->entry.precnode->precord;
    return PyString_FromString(prec->name);
}

static PyObject* pyRecord_info(pyRecord *self, PyObject *args)
{
    const char *name;
    PyObject *def = Py_None;
    DBENTRY entry;

    if(!PyArg_ParseTuple(args, "s|O", &name, &def))
        return NULL;

    dbCopyEntryContents(&self->entry, &entry);

    if(dbFindInfo(&entry, name)) {
        if(def) {
            Py_INCREF(def);
            return def;
        } else {
            PyErr_SetNone(PyExc_KeyError);
            return NULL;
        }
    }

    return PyString_FromString(dbGetInfoString(&entry));
}

static PyObject* pyRecord_infos(pyRecord *self)
{
    PyObject *dict;
    DBENTRY entry;
    long status;

    dict = PyDict_New();
    if(!dict)
        return NULL;

    dbCopyEntryContents(&self->entry, &entry);

    for(status = dbFirstInfo(&entry); status==0;  status = dbNextInfo(&entry))
    {
        PyObject *val = PyString_FromString(dbGetInfoString(&entry));
        if(!val)
            goto fail;

        if(PyDict_SetItemString(dict, dbGetInfoName(&entry), val)<0) {
            Py_DECREF(val);
            goto fail;
        }
    }

    return dict;
fail:
    Py_DECREF(dict);
    return NULL;
}

static PyObject* pyRecord_scan(pyRecord *self, PyObject *args, PyObject *kws)
{
    dbCommon *prec = self->entry.precnode->precord;

    static char* names[] = {"sync", "reason", NULL};
    PyObject *reason = Py_None;
    PyObject *sync = Py_False;

    if(!PyArg_ParseTupleAndKeywords(args, kws, "|OO", names, &sync, &reason))
        return NULL;

    if(!PyObject_IsTrue(sync)) {
        scanOnce(prec);
        Py_RETURN_NONE;

    } else {
        long ret;
        setReasonPyRecord(prec, reason);

        Py_BEGIN_ALLOW_THREADS {

            dbScanLock(prec);
            ret = dbProcess(prec);
            dbScanUnlock(prec);

        } Py_END_ALLOW_THREADS

        clearReasonPyRecord(prec);

        return PyLong_FromLong(ret);
    }
}

static PyObject *pyRecord_asyncStart(pyRecord *self)
{
    dbCommon *prec=self->entry.precnode->precord;
    epicsUInt8 pact = prec->pact;
    if(!isPyRecord(prec)) {
        PyErr_SetString(PyExc_RuntimeError, "Not a Python Device record");
        return NULL;
    }
    prec->pact = 1;
    return PyLong_FromLong(pact);
}

static PyObject *pyRecord_asyncFinish(pyRecord *self, PyObject *args, PyObject *kws)
{
    long pact, ret;
    dbCommon *prec = self->entry.precnode->precord;

    static char* names[] = {"reason", NULL};
    PyObject *reason = Py_None;

    if(!PyArg_ParseTupleAndKeywords(args, kws, "|O", names, &reason))
        return NULL;

    if(!isPyRecord(prec)) {
        PyErr_SetString(PyExc_RuntimeError, "Not a Python Device record");
        return NULL;
    }

    Py_INCREF(self);

    setReasonPyRecord(prec, reason);

    Py_BEGIN_ALLOW_THREADS {
        rset *rsup = prec->rset;

        dbScanLock(prec);
        pact = prec->pact;
        if(pact) {
            ret = (*rsup->process)(prec);
            /* Out devsup always clears PACT if initially set */
        }
        dbScanUnlock(prec);

    } Py_END_ALLOW_THREADS

    clearReasonPyRecord(prec);

    if(!pact) {
        PyErr_SetString(PyExc_ValueError, "Python Device record was not active");
        return NULL;
    }

    Py_DECREF(self);

    return PyLong_FromLong(ret);
}

static PyMethodDef pyRecord_methods[] = {
    {"name", (PyCFunction)pyRecord_name, METH_NOARGS,
     "Return record name string"},
    {"info", (PyCFunction)pyRecord_info, METH_VARARGS,
     "Lookup info name\ninfo(name, def=None)"},
    {"infos", (PyCFunction)pyRecord_infos, METH_NOARGS,
     "Return a dictionary of all infos for this record."},
    {"scan", (PyCFunction)pyRecord_scan, METH_VARARGS|METH_KEYWORDS,
     "scan(sync=False)\nScan this record.  If sync is False then"
     "a scan request is queued.  If sync is True then the record"
     "is scannined immidately on the current thread."},
    {"asyncStart", (PyCFunction)pyRecord_asyncStart, METH_NOARGS,
     "Begin an asynchronous action.  Record must be locked!"},
    {"asyncFinish", (PyCFunction)pyRecord_asyncFinish, METH_VARARGS|METH_KEYWORDS,
     "Complete an asynchronous action.  Record must *not* be locked!"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject pyRecord_type = {
    PyObject_HEAD_INIT(NULL)
    0,
    "_dbapi._Record",
    sizeof(pyRecord),
};


int pyRecord_prepare(void)
{
    pyRecord_type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    pyRecord_type.tp_methods = pyRecord_methods;

    pyRecord_type.tp_new = (newfunc)pyRecord_new;
    pyRecord_type.tp_dealloc = (destructor)pyRecord_dealloc;
    pyRecord_type.tp_init = (initproc)pyRecord_Init;
    pyRecord_type.tp_compare = (cmpfunc)pyRecord_compare;

    if(PyType_Ready(&pyRecord_type)<0)
        return -1;
    return 0;
}

void pyRecord_setup(PyObject *module)
{
    PyObject *typeobj=(PyObject*)&pyRecord_type;
    Py_INCREF(typeobj);
    PyModule_AddObject(module, "_Record", (PyObject*)&pyRecord_type);
}
