
/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

#include <epicsVersion.h>
#include <dbCommon.h>
#include <dbAccess.h>
#include <dbStaticLib.h>
#include <dbScan.h>

typedef struct {
    PyObject_HEAD

    DBENTRY entry;
} pyRecord;

static int pyRecord_Init(pyRecord *self, PyObject *args, PyObject *kws)
{
    const char *recname;

    if(!PyArg_ParseTuple(args, "s", &recname))
        return -1;

    dbInitEntry(pdbbase, &self->entry);

    if(dbFindRecord(&self->entry, recname)){
        PyErr_SetString(PyExc_ValueError, "Record not found");
        return -1;
    }
    return 0;
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

    static char* names[] = {"sync", NULL};
    PyObject *sync = Py_False;

    if(!PyArg_ParseTupleAndKeywords(args, kws, "|O", names, &sync))
        return NULL;

    if(!PyObject_IsTrue(sync)) {
        scanOnce(prec);
    } else {
        Py_BEGIN_ALLOW_THREADS {

            dbScanLock(prec);
            dbProcess(prec);
            dbScanUnlock(prec);

        } Py_END_ALLOW_THREADS
    }

    Py_RETURN_NONE;
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
     "is scannined immidately on the current thread."
    },
    {NULL, NULL, 0, NULL}
};

static PyTypeObject pyRecord_type = {
    PyObject_HEAD_INIT(NULL)
    0,
    "_dbapi.Record",
    sizeof(pyRecord),
};


int pyRecord_prepare(void)
{
    pyRecord_type.tp_flags = Py_TPFLAGS_DEFAULT;
    pyRecord_type.tp_methods = pyRecord_methods;
    pyRecord_type.tp_init = (initproc)pyRecord_Init;

    pyRecord_type.tp_new = PyType_GenericNew;
    if(PyType_Ready(&pyRecord_type)<0)
        return -1;
    return 0;
}

void pyRecord_setup(PyObject *module)
{
    PyObject *typeobj=(PyObject*)&pyRecord_type;
    Py_INCREF(typeobj);
    PyModule_AddObject(module, "Field", (PyObject*)&pyRecord_type);
}
