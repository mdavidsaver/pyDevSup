/* Global interpreter setup
 */

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

#include "pydevsup.h"

typedef struct {
    const initHookState state;
    const char * const name;
} pystate;

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

    state = PyGILState_Ensure();

    if(PyRun_SimpleStringFlags(code, NULL)!=0)
        PyErr_Print();

    PyGILState_Release(state);
}

void pyfile(const char* file)
{
    FILE *fp;
    PyGILState_STATE state;

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

static void pyhook(initHookState state)
{
    static int madenoise = 0;
    PyGILState_STATE gilstate;
    PyObject *mod, *ret;

    /* ignore deprecated init hooks */
    if(state==initHookAfterInterruptAccept || state==initHookAtEnd)
        return;

    gilstate = PyGILState_Ensure();

    mod = PyImport_ImportModule("devsup.hooks");
    if(!mod) {
        if(!madenoise)
            fprintf(stderr, "Couldn't import devsup.hooks\n");
        madenoise=1;
        return;
    }
    ret = PyObject_CallMethod(mod, "_runhook", "l", (long)state);
    Py_DECREF(mod);
    if(PyErr_Occurred()) {
        PyErr_Print();
        PyErr_Clear();
    }
    Py_XDECREF(ret);

    PyGILState_Release(gilstate);
}

static const char sitestr[] = EPICS_SITE_VERSION;

static PyObject *modversion(PyObject *self)
{
    int ver=EPICS_VERSION, rev=EPICS_REVISION, mod=EPICS_MODIFICATION, patch=EPICS_PATCH_LEVEL;
    return Py_BuildValue("iiiis", ver, rev, mod, patch, sitestr);
}

static PyMethodDef devsup_methods[] = {
    {"verinfo", (PyCFunction)modversion, METH_NOARGS,
     "EPICS Version information\nreturn (MAJOR, MINOR, MOD, PATH, \"site\""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef dbapimodule = {
  PyModuleDef_HEAD_INIT,
    "_dbapi",
    NULL,
    -1,
    devsup_methods
};
#endif

/* initialize "magic" builtin module */
PyMODINIT_FUNC init_dbapi(void)
{
    PyObject *mod = NULL, *hookdict;
    pystate *st;

    import_array();

#if PY_MAJOR_VERSION >= 3
    mod = PyModule_Create(&dbapimodule);
#else
    mod = Py_InitModule("_dbapi", devsup_methods);
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

    if(pyField_prepare(mod))
        goto fail;
    if(pyRecord_prepare(mod))
        goto fail;

    MODINIT_RET(mod);

fail:
    fprintf(stderr, "Failed to initialize builtin _dbapi module!\n");
    Py_XDECREF(mod);
    MODINIT_RET(NULL);
}


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef constantsmodule = {
  PyModuleDef_HEAD_INIT,
    "_dbconstants",
    NULL,
    -1,
    NULL
};
#endif

/* initialize "magic" builtin module */
PyMODINIT_FUNC init_dbconstants(void)
{
    PyObject *mod = NULL;

#if PY_MAJOR_VERSION >= 3
    mod = PyModule_Create(&constantsmodule);
#else
    mod = Py_InitModule("_dbconstants", NULL);
#endif
    if(!mod)
        MODINIT_RET(NULL);

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

    MODINIT_RET(mod);
}

static void cleanupPy(void *junk)
{
    PyThreadState *state = PyGILState_GetThisThreadState();

    PyEval_RestoreThread(state);

    /* special "fake" hook for shutdown */
    pyhook((initHookState)9999);

    pyDBD_cleanup();

    pyField_cleanup();

    Py_Finalize();
}

/* Initialize the interpreter environment
 */
static void setupPyInit(void)
{
    PyThreadState *state;

    PyImport_AppendInittab("_dbapi", init_dbapi);
    PyImport_AppendInittab("_dbconstants", init_dbconstants);

    Py_Initialize();
    PyEval_InitThreads();

    state = PyEval_SaveThread();

    epicsAtExit(&cleanupPy, NULL);
}

#include <iocsh.h>

static const iocshArg argCode = {"python code", iocshArgString};
static const iocshArg argFile = {"file", iocshArgString};

static const iocshArg* const codeArgs[] = {&argCode};
static const iocshArg* const fileArgs[] = {&argFile};

static const iocshFuncDef codeDef = {"py", 1, codeArgs};
static const iocshFuncDef fileDef = {"pyfile", 1, fileArgs};

static void codeRun(const iocshArgBuf *args){py(args[0].sval);}
static void fileRun(const iocshArgBuf *args){pyfile(args[0].sval);}

static void pySetupReg(void)
{
    PyGILState_STATE state;

    setupPyInit();
    iocshRegister(&codeDef, &codeRun);
    iocshRegister(&fileDef, &fileRun);
    initHookRegister(&pyhook);

    state = PyGILState_Ensure();
    init_dbapi();
    if(PyErr_Occurred()) {
        PyErr_Print();
        PyErr_Clear();
    }
    PyGILState_Release(state);
}

#include <epicsExport.h>
epicsExportRegistrar(pySetupReg);
