/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>
#ifdef HAVE_NUMPY
#include <numpy/ndarrayobject.h>
#endif

#include <stdio.h>

#include <epicsVersion.h>
#include <dbCommon.h>
#include <dbAccess.h>
#include <dbStaticLib.h>
#include <dbScan.h>
#include <initHooks.h>
#include <epicsThread.h>
#include <epicsExit.h>
#include <alarm.h>
#include <iocsh.h>
#include <iocInit.h>

#include "pydevsup.h"

initHookState pyInitLastState;

extern int pyDevSupCommon_registerRecordDeviceDriver(DBBASE *pbase);

typedef struct {
    const initHookState state;
    const char * const name;
} pystate;

epicsThreadPrivateId pyDevReasonID;

#define INITST(hook) {initHook ## hook, #hook }
static pystate statenames[] = {
    INITST(AtIocBuild),

    INITST(AtBeginning),
    INITST(AfterCallbackInit),
    INITST(AfterCaLinkInit),
    INITST(AfterInitDrvSup),
    INITST(AfterInitRecSup),
    INITST(AfterInitDevSup),
    INITST(AfterInitDatabase),
    INITST(AfterFinishDevSup),
    INITST(AfterScanInit),

    INITST(AfterInitialProcess),
    INITST(AfterCaServerInit),
    INITST(AfterIocBuilt),
    INITST(AtIocRun),
    INITST(AfterDatabaseRunning),
    INITST(AfterCaServerRunning),
    INITST(AfterIocRunning),
    INITST(AtIocPause),

    INITST(AfterCaServerPaused),
    INITST(AfterDatabasePaused),
    INITST(AfterIocPaused),

    {(initHookState)9999, "AtIocExit"},

    {(initHookState)0, NULL}
};
#undef INITST

void py(const char* code)
{
    PyGILState_STATE state;

    if(!code)
        return;

    state = PyGILState_Ensure();

    if(PyRun_SimpleStringFlags(code, NULL)!=0)
        PyErr_Print();

    PyGILState_Release(state);
}

void pyfile(const char* file)
{
    FILE *fp;
    PyGILState_STATE state;

    if(!file)
        return;

    state = PyGILState_Ensure();

    fp = fopen(file, "r");
    if(!fp) {
        fprintf(stderr, "Failed to open: %s\n", file);
        perror("open");
    } else {
        if(PyRun_SimpleFileExFlags(fp, file, 1, NULL)!=0)
            PyErr_Print();
    }
    /* fp closed by python */

    PyGILState_Release(state);
}



static const iocshArg argCode = {"python code", iocshArgString};
static const iocshArg argFile = {"file", iocshArgString};

static const iocshArg* const codeArgs[] = {&argCode};
static const iocshArg* const fileArgs[] = {&argFile};

static const iocshFuncDef codeDef = {"py", 1, codeArgs};
static const iocshFuncDef fileDef = {"pyfile", 1, fileArgs};

static void codeRun(const iocshArgBuf *args){py(args[0].sval);}
static void fileRun(const iocshArgBuf *args){pyfile(args[0].sval);}

initHookState pyInitLastState = (initHookState)-1;

static void pyhook(initHookState state)
{
    static int madenoise = 0;
    PyGILState_STATE gilstate;
    PyObject *mod, *ret;

    /* ignore deprecated init hooks */
    if(state==initHookAfterInterruptAccept || state==initHookAtEnd)
        return;

    gilstate = PyGILState_Ensure();

    pyInitLastState = state;

    mod = PyImport_ImportModule("devsup.hooks");
    if(!mod) {
        if(!madenoise)
            fprintf(stderr, "Error: Couldn't import devsup.hooks!  Python module initHooks can not be run!\n");
        madenoise=1;
        goto fail;
    }
    ret = PyObject_CallMethod(mod, "_runhook", "l", (long)state);
    Py_DECREF(mod);
    if(PyErr_Occurred()) {
        PyErr_Print();
        PyErr_Clear();
    }
    Py_XDECREF(ret);

fail:
    PyGILState_Release(gilstate);
}

static
PyObject* py_announce(PyObject *unused, PyObject *args, PyObject *kws)
{
    static char* names[] = {"state", NULL};
    int state;
    if(!PyArg_ParseTupleAndKeywords(args, kws, "i", names, &state))
        return NULL;

    Py_BEGIN_ALLOW_THREADS {
        initHookAnnounce((initHookState)state);
    } Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

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
        if(fname) {
            status = dbReadDatabase(&pdbbase, fname, path, sub);
        } else {
            FILE *ff = fdopen(fd, "r");
            status = dbReadDatabaseFP(&pdbbase, ff, path, sub);
            // dbReadDatabaseFP() has called fclose()
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
PyObject *py_iocInit(PyObject *unused, PyObject *args, PyObject *kws)
{
    static char* names[] = {"isolate", NULL};
    PyObject *pyisolate = Py_True;
    int isolate, ret;
    if(!PyArg_ParseTupleAndKeywords(args, kws, "|O", names, &pyisolate))
        return NULL;

    isolate = PyObject_IsTrue(pyisolate);
    Py_BEGIN_ALLOW_THREADS {
        ret = isolate ? iocBuildIsolated() : iocBuild();
        if(!ret)
            ret = iocRun();
    } Py_END_ALLOW_THREADS

    if(ret)
        return PyErr_Format(PyExc_RuntimeError, "Error %d", ret);

    Py_RETURN_NONE;
}

static
PyObject *py_pyDevSupCommon(PyObject *unused)
{
    Py_BEGIN_ALLOW_THREADS {
        pyDevSupCommon_registerRecordDeviceDriver(pdbbase);
    } Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static struct PyMethodDef dbapimethod[] = {
    {"initHookAnnounce", (PyCFunction)py_announce, METH_VARARGS|METH_KEYWORDS,
     "initHookAnnounce(state)\n"},
    {"iocsh", (PyCFunction)py_iocsh, METH_VARARGS|METH_KEYWORDS,
     "Execute IOC shell script or command"},
    {"dbReadDatabase", (PyCFunction)py_dbReadDatabase, METH_VARARGS|METH_KEYWORDS,
     "Load EPICS database file"},
    {"iocInit", (PyCFunction)py_iocInit, METH_NOARGS,
     "Initialize IOC"},
    {"_dbd_setup", (PyCFunction)pyDBD_setup, METH_NOARGS, ""},
    {"_dbd_rrd_base", (PyCFunction)py_pyDevSupCommon, METH_NOARGS, ""},
    {"_dbd_cleanup", (PyCFunction)pyDBD_cleanup, METH_NOARGS, ""},
    {NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef dbapimodule = {
  PyModuleDef_HEAD_INIT,
    "devsup._dbapi",
    NULL,
    -1,
    &dbapimethod
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__dbapi(void)
#else
PyMODINIT_FUNC init_dbapi(void)
#endif
{
    PyObject *mod = NULL, *hookdict, *vertup;
    pystate *st;

    pyDevReasonID = epicsThreadPrivateCreate();

    iocshRegister(&codeDef, &codeRun);
    iocshRegister(&fileDef, &fileRun);
    initHookRegister(&pyhook);

    import_array();

#if PY_MAJOR_VERSION >= 3
    mod = PyModule_Create(&dbapimodule);
#else
    mod = Py_InitModule("devsup._dbapi", dbapimethod);
#endif
    if(!mod)
        goto fail;

    hookdict = PyDict_New();
    if(!hookdict)
        goto fail;

    PyModule_AddObject(mod, "_hooks", hookdict);

    for(st = statenames; st->name; st++) {
        PyObject *ent = PyInt_FromLong((long)st->state);
        if(!ent) goto fail;
        if(PyDict_SetItemString(hookdict, st->name, ent)) {
            Py_DECREF(ent);
            goto fail;
        }
    }

    PyModule_AddIntMacro(mod, NO_ALARM);
    PyModule_AddIntMacro(mod, MINOR_ALARM);
    PyModule_AddIntMacro(mod, MAJOR_ALARM);
    PyModule_AddIntMacro(mod, INVALID_ALARM);

    PyModule_AddIntMacro(mod, READ_ALARM);
    PyModule_AddIntMacro(mod, WRITE_ALARM);
    PyModule_AddIntMacro(mod, HIHI_ALARM);
    PyModule_AddIntMacro(mod, HIGH_ALARM);
    PyModule_AddIntMacro(mod, LOLO_ALARM);
    PyModule_AddIntMacro(mod, LOW_ALARM);
    PyModule_AddIntMacro(mod, STATE_ALARM);
    PyModule_AddIntMacro(mod, COS_ALARM);
    PyModule_AddIntMacro(mod, COMM_ALARM);
    PyModule_AddIntMacro(mod, TIMEOUT_ALARM);
    PyModule_AddIntMacro(mod, HW_LIMIT_ALARM);
    PyModule_AddIntMacro(mod, CALC_ALARM);
    PyModule_AddIntMacro(mod, SCAN_ALARM);
    PyModule_AddIntMacro(mod, LINK_ALARM);
    PyModule_AddIntMacro(mod, SOFT_ALARM);
    PyModule_AddIntMacro(mod, BAD_SUB_ALARM);
    PyModule_AddIntMacro(mod, UDF_ALARM);
    PyModule_AddIntMacro(mod, DISABLE_ALARM);
    PyModule_AddIntMacro(mod, SIMM_ALARM);
    PyModule_AddIntMacro(mod, READ_ACCESS_ALARM);
    PyModule_AddIntMacro(mod, WRITE_ACCESS_ALARM);

    /* standard macros from epicsVersion.h */
    PyModule_AddStringMacro(mod, EPICS_VERSION_STRING);
#ifdef EPICS_DEV_SNAPSHOT
    PyModule_AddStringMacro(mod, EPICS_DEV_SNAPSHOT);
#endif
    PyModule_AddStringMacro(mod, EPICS_SITE_VERSION);
    PyModule_AddIntMacro(mod, EPICS_VERSION);
    PyModule_AddIntMacro(mod, EPICS_REVISION);
    PyModule_AddIntMacro(mod, EPICS_PATCH_LEVEL);
    PyModule_AddIntMacro(mod, EPICS_MODIFICATION);
    /* additional build time info */
    PyModule_AddStringMacro(mod, XEPICS_ARCH);
    PyModule_AddStringMacro(mod, XPYDEV_BASE);
    PyModule_AddStringMacro(mod, XEPICS_BASE);


    vertup = Py_BuildValue("(iiiiss)",
                           (int)EPICS_VERSION,
                           (int)EPICS_REVISION,
                           (int)EPICS_MODIFICATION,
                           (int)EPICS_PATCH_LEVEL,
                           EPICS_SITE_VERSION,
#ifdef EPICS_DEV_SNAPSHOT
                           EPICS_DEV_SNAPSHOT);
#else
                           "");
#endif
    if(vertup)
        PyModule_AddObject(mod, "epicsver", vertup);

    /* pyDevSup version */
    vertup = Py_BuildValue("(ii)", 0, 2);
    if(vertup)
        PyModule_AddObject(mod, "pydevver", vertup);

    if(pyField_prepare(mod))
        goto fail;
    if(pyRecord_prepare(mod))
        goto fail;
    if(pyUTest_prepare(mod))
        goto fail;

    MODINIT_RET(mod);

fail:
    fprintf(stderr, "Failed to initialize builtin _dbapi module!\n");
    Py_XDECREF(mod);
    MODINIT_RET(NULL);
}
