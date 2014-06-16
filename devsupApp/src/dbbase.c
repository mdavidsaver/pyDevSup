
/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

#include <epicsVersion.h>
#include <dbCommon.h>
#include <dbStaticLib.h>
#include <dbAccess.h>
#include <initHooks.h>
#include <iocsh.h>
#include <iocInit.h>

#include "pydevsup.h"

static
PyObject *py_iocsh(PyObject *unused, PyObject *args, PyObject *kws)
{
    int ret;
    static char* names[] = {"script", "cmd", NULL};
    char *script=NULL, *cmd=NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kws, "|ss", names, &script, &cmd))
        return NULL;

    if(!(!script ^ !cmd)) {
        PyErr_SetString(PyExc_ValueError, "iocsh requires a script file name or command string");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS {
        if(script)
            ret = iocsh(script);
        else
            ret = iocshCmd(cmd);
    } Py_END_ALLOW_THREADS

    return PyInt_FromLong(ret);
}

static
PyObject *py_dbReadDatabase(PyObject *unused, PyObject *args, PyObject *kws)
{
    long status;
    static char* names[] = {"name", "fp", "path", "sub", NULL};
    char *fname=NULL, *path=NULL, *sub=NULL;
    int fd=-1;

    if(!PyArg_ParseTupleAndKeywords(args, kws, "|siss", names, &fname, &fd, &path, &sub))
        return NULL;

    if(!((!fname) ^ (fd<0))) {
        PyErr_SetString(PyExc_ValueError, "dbReadDatabase requires a file name or descriptor");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS {
        if(fname)
            status = dbReadDatabase(&pdbbase, fname, path, sub);
        else {
            FILE *ff = fdopen(fd, "r");
            status = dbReadDatabaseFP(&pdbbase, ff, path, sub);
            fclose(ff);
        }
    } Py_END_ALLOW_THREADS

    if(status) {
        char buf[30];
        errSymLookup(status, buf, sizeof(buf));
        PyErr_SetString(PyExc_RuntimeError, buf);
        return NULL;
    }
    Py_RETURN_NONE;
}

static
PyObject *py_iocInit(PyObject *unused)
{
    Py_BEGIN_ALLOW_THREADS {
        iocInit();
    } Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static struct PyMethodDef dbbasemethods[] = {
    {"iocsh", (PyCFunction)py_iocsh, METH_VARARGS|METH_KEYWORDS,
     "Execute IOC shell script or command"},
    {"dbReadDatabase", (PyCFunction)py_dbReadDatabase, METH_VARARGS|METH_KEYWORDS,
     "Load EPICS database file"},
    {"iocInit", (PyCFunction)py_iocInit, METH_NOARGS,
     "Initialize IOC"},
    {NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef dbbasemodule = {
  PyModuleDef_HEAD_INIT,
    "_dbbase",
    NULL,
    -1,
    &dbbasemethods
};
#endif

/* initialize "magic" builtin module */
PyMODINIT_FUNC init_dbbase(void)
{
    PyObject *mod = NULL, *obj = NULL;

#if PY_MAJOR_VERSION >= 3
    mod = PyModule_Create(&dbbasemodule);
#else
    mod = Py_InitModule("_dbbase", dbbasemethods);
#endif
    if(!mod)
        goto fail;

#if PY_MAJOR_VERSION >= 3 || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION>=7)
    obj = PyCapsule_New(pdbbase, "pdbbase", NULL);
#else
    obj = PyCObject_FromVoidPtrAndDesc(pdbbase, "pdbbase", NULL);
#endif
    if(!obj)
        goto fail;
    PyModule_AddObject(mod, "pdbbase", obj);

    MODINIT_RET(mod);
fail:
    Py_XDECREF(obj);
    Py_XDECREF(mod);
    fprintf(stderr, "Failed to initialize builtin _dbbase module!\n");
    MODINIT_RET(NULL);
}
