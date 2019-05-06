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

static void cleanupPy(void *junk)
{
    /* safe because exit hooks are only run once, from a single thread */
    static int done;

    if(done) return;
    done = 1;

    PyThreadState *state = PyGILState_GetThisThreadState();

    PyEval_RestoreThread(state);

    if(PyRun_SimpleString("import devsup\n"
                          "devsup._fini(iocMain=True)\n"
    )) {
        PyErr_Print();
        PyErr_Clear();
    }

    Py_Finalize(); // calls python atexit hooks
}

static void extendPath(PyObject *list,
                       const char *base,
                       const char *archdir)
{
    PyObject *mod, *ret;

    mod = PyImport_ImportModule("os.path");
    if(!mod)
        return;

    ret = PyObject_CallMethod(mod, "join", "sss", base, PYDIR, archdir);
    if(ret && !PySequence_Contains(list, ret)) {
        PyList_Insert(list, 0, ret);
    }
    Py_XDECREF(ret);
    Py_DECREF(mod);
    if(PyErr_Occurred()) {
        PyErr_Print();
        PyErr_Clear();
    }
}

static void insertDefaultPath(PyObject *list)
{
    const char *basedir, *pydevdir, *top, *arch;

    basedir = getenv("EPICS_BASE");
    if(!basedir)
        basedir = XEPICS_BASE;
    pydevdir = getenv("PYDEV_BASE");
    if(!pydevdir)
        pydevdir = XPYDEV_BASE;
    top = getenv("TOP");
    arch = getenv("ARCH");
    if(!arch)
        arch = XEPICS_ARCH;

    assert(PyList_Check(list));
    assert(PySequence_Check(list));
    extendPath(list, basedir, arch);
    extendPath(list, pydevdir, arch);
    if(top)
        extendPath(list, top, arch);
}

static void setupPyPath(void)
{
    PyObject *mod, *path = NULL;

    mod = PyImport_ImportModule("sys");
    if(mod)
        path = PyObject_GetAttrString(mod, "path");
    Py_XDECREF(mod);

    if(path) {
        PyObject *cur;
        char cwd[PATH_MAX];

        insertDefaultPath(path);

        /* prepend current directory */
        if(getcwd(cwd, sizeof(cwd)-1)) {
            cwd[sizeof(cwd)-1] = '\0';
            cur = PyString_FromString(cwd);
            if(cur)
                PyList_Insert(path, 0, cur);
            Py_XDECREF(cur);
        }
    }
    Py_XDECREF(path);
}

static void cleanupPrep(initHookState state)
{
    /* register a second time to better our chances of running
     * first on exit.  eg. before cacExitHandler()
     */
    if(state==initHookAfterInitDevSup)
        epicsAtExit(&cleanupPy, NULL);
}

static void pySetupReg(void)
{
    Py_InitializeEx(0);
    PyEval_InitThreads();

    setupPyPath();

    if(PyRun_SimpleString("import devsup\n"
                          "devsup._init(iocMain=True)\n"
    )) {
        PyErr_Print();
        PyErr_Clear();
    }

    (void)PyEval_SaveThread();

    /* register first time to ensure cleanupPy is run at least once */
    epicsAtExit(&cleanupPy, NULL);

    initHookRegister(&cleanupPrep);
}

#include <epicsExport.h>
epicsExportRegistrar(pySetupReg);
