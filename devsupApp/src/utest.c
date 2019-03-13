
/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

#include <epicsVersion.h>
#include <dbEvent.h>
#include <iocInit.h>
#include <errlog.h>

#if EPICS_VERSION>3 || (EPICS_VERSION==3 && EPICS_REVISION>=15)
#  include <dbUnitTest.h>
#  define HAVE_DBTEST
#endif

#include "pydevsup.h"

#ifdef HAVE_DBTEST
static dbEventCtx testEvtCtx;
#endif

typedef struct {
    PyObject_HEAD
} UTest;

static PyTypeObject UTest_type = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,
#endif
    "_dbapi._UTest",
    sizeof(UTest),
};

static PyObject* utest_prepare(PyObject *unused)
{
    Py_BEGIN_ALLOW_THREADS {
        //testdbPrepare(); doesn't do anything essential for us as of 7.0.2
    } Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* utest_init(PyObject *unused)
{
#ifdef HAVE_DBTEST
    int ret;
    if(testEvtCtx)
        return PyErr_Format(PyExc_RuntimeError, "Missing testIocShutdownOk()");

    // like, testIocInitOk() without testAbort()
    Py_BEGIN_ALLOW_THREADS {
        eltc(0);
        ret = iocBuildIsolated() || iocRun();
        eltc(1);
    } Py_END_ALLOW_THREADS
    if(ret) {
        return PyErr_Format(PyExc_RuntimeError, "iocInit fails with %d", ret);
    }

    Py_BEGIN_ALLOW_THREADS {
        testEvtCtx=db_init_events();
    } Py_END_ALLOW_THREADS
    if(!testEvtCtx) {
        iocShutdown();
        return PyErr_Format(PyExc_RuntimeError, "iocInit fails create dbEvent context");
    }

    Py_BEGIN_ALLOW_THREADS {
        ret = db_start_events(testEvtCtx, "CAS-test-py", NULL, NULL, epicsThreadPriorityCAServerLow);
    } Py_END_ALLOW_THREADS
    if(ret!=DB_EVENT_OK) {
        db_close_events(testEvtCtx);
        testEvtCtx = NULL;
        iocShutdown();
        return PyErr_Format(PyExc_RuntimeError, "db_start_events fails with %d", ret);
    }
    Py_RETURN_NONE;
#else
    return PyErr_Format(PyExc_RuntimeError, "Requires Base >=3.15");
#endif
}

static PyObject* utest_shutdown(PyObject *unused)
{
#ifdef HAVE_DBTEST
    Py_BEGIN_ALLOW_THREADS {
        //testIocShutdownOk();
        db_close_events(testEvtCtx);
        testEvtCtx = NULL;
        iocShutdown();
    } Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
#else
    return PyErr_Format(PyExc_RuntimeError, "Requires Base >=3.15");
#endif
}

static PyObject* utest_cleanup(PyObject *unused)
{
#ifdef HAVE_DBTEST
    Py_BEGIN_ALLOW_THREADS {
        testdbCleanup();
        errlogFlush();
    } Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
#else
    return PyErr_Format(PyExc_RuntimeError, "Requires Base >=3.15");
#endif
}

static PyMethodDef UTest_methods[] = {
    {"testdbPrepare", (PyCFunction)&utest_prepare, METH_STATIC|METH_NOARGS, ""},
    {"testIocInitOk", (PyCFunction)&utest_init, METH_STATIC|METH_NOARGS, ""},
    {"testIocShutdownOk", (PyCFunction)&utest_shutdown, METH_STATIC|METH_NOARGS, ""},
    {"testdbCleanup", (PyCFunction)&utest_cleanup, METH_STATIC|METH_NOARGS, ""},
    {NULL}
};

int pyUTest_prepare(PyObject *module)
{
    PyObject *typeobj=(PyObject*)&UTest_type;

    UTest_type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    UTest_type.tp_methods = UTest_methods;

    if(PyType_Ready(&UTest_type)<0)
        return -1;

    Py_INCREF(typeobj);
    if(PyModule_AddObject(module, "_UTest", typeobj)) {
        Py_DECREF(typeobj);
        return -1;
    }

    return 0;
}
