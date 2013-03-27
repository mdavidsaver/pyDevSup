/* Global interpreter setup
 */

/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

#include <stdio.h>

#include <epicsVersion.h>
#include <dbCommon.h>
#include <dbAccess.h>
#include <dbStaticLib.h>
#include <dbScan.h>
#include <initHooks.h>
#include <epicsThread.h>
#include <epicsExit.h>

/* dictionary of initHook names */
static PyObject *hooktable;

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
    {(initHookState)0, NULL}
};
#undef INITST

static void cleanupPy(void *junk)
{
    PyThreadState *state = PyGILState_GetThisThreadState();

    PyEval_RestoreThread(state);

    /* release extra reference for hooktable */
    Py_DECREF(hooktable);
    hooktable = NULL;

    Py_Finalize();
}

/* Initialize the interpreter environment
 */
static epicsThreadOnceId setupPyOnceId = EPICS_THREAD_ONCE_INIT;
static void setupPyOnce(void *junk)
{
    PyThreadState *state;

    Py_Initialize();
    PyEval_InitThreads();

    state = PyEval_SaveThread();

    epicsAtExit(&cleanupPy, NULL);
}

void evalPy(const char* code)
{
    PyGILState_STATE state;

    state = PyGILState_Ensure();

    if(PyRun_SimpleStringFlags(code, NULL)!=0)
        PyErr_Print();

    PyGILState_Release(state);
}

void evalFilePy(const char* file)
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
    PyGILState_STATE gilstate;

    gilstate = PyGILState_Ensure();

    if(hooktable && PyDict_Check(hooktable)) {
        PyObject *next;
        PyObject *key = PyInt_FromLong((long)state);
        PyObject *list = PyDict_GetItem(hooktable, key);
        Py_DECREF(key);

        if(list) {

            list = PyObject_GetIter(list);
            if(!list) {
                fprintf(stderr, "hook sequence not iterable!");

            } else {

                while((next=PyIter_Next(list))!=NULL) {
                    PyObject *obj;
                    if(!PyCallable_Check(next))
                        continue;
                    obj = PyObject_CallFunction(next, "");
                    Py_DECREF(next);
                    if(obj)
                        Py_DECREF(obj);
                    else {
                        PyErr_Print();
                        PyErr_Clear();
                    }
                }
                if(!PyErr_Occurred()) {
                    PyErr_Print();
                    PyErr_Clear();
                }

                Py_DECREF(list);
            }
        }
    }

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

int pyField_prepare(void);
void pyField_setup(PyObject *module);

int pyRecord_prepare(void);
void pyRecord_setup(PyObject *module);

/* initialize "magic" builtin module */
static void init_dbapi(void)
{
    PyObject *mod, *hookdict, *pysuptable;
    pystate *st;
    PyGILState_STATE state;

    state = PyGILState_Ensure();

    hooktable = PyDict_New();
    if(!hooktable)
        return;

    if(pyField_prepare())
        return;
    if(pyRecord_prepare())
        return;

    mod = Py_InitModule("_dbapi", devsup_methods);

    pysuptable = PySet_New(NULL);
    if(!pysuptable)
        return;
    PyModule_AddObject(mod, "_supports", pysuptable);

    hookdict = PyDict_New();
    if(!hookdict)
        return;
    PyModule_AddObject(mod, "_hooks", hookdict);

    for(st = statenames; st->name; st++) {
        PyDict_SetItemString(hookdict, st->name, PyInt_FromLong((long)st->state));
    }
    Py_INCREF(hooktable); /* an extra ref for the global pointer */
    PyModule_AddObject(mod, "_hooktable", hooktable);

    pyField_setup(mod);
    pyRecord_setup(mod);

    PyGILState_Release(state);
}

#include <iocsh.h>

static const iocshArg argCode = {"python code", iocshArgString};
static const iocshArg argFile = {"file", iocshArgString};

static const iocshArg* const codeArgs[] = {&argCode};
static const iocshArg* const fileArgs[] = {&argFile};

static const iocshFuncDef codeDef = {"evalPy", 1, codeArgs};
static const iocshFuncDef fileDef = {"evalFilePy", 1, fileArgs};

static void codeRun(const iocshArgBuf *args){evalPy(args[0].sval);}
static void fileRun(const iocshArgBuf *args){evalFilePy(args[0].sval);}

static void pySetupReg(void)
{
    epicsThreadOnce(&setupPyOnceId, &setupPyOnce, NULL);
    iocshRegister(&codeDef, &codeRun);
    iocshRegister(&fileDef, &fileRun);
    initHookRegister(&pyhook);
    init_dbapi();
}

#include <epicsExport.h>
epicsExportRegistrar(pySetupReg);
