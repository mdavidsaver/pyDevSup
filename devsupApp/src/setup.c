/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

#include <stdio.h>

#include <epicsThread.h>
#include <epicsExit.h>
#include <initHooks.h>

static PyThreadState *main_state;

static void cleanupPy(void *junk)
{
    PyEval_RestoreThread(main_state);

    Py_Finalize();
}

/* Initialize the interpreter environment
 */
static void setupPyOnce(void *junk)
{
    Py_Initialize();

    PyEval_InitThreads();

    epicsAtExit(&cleanupPy, NULL);

    main_state = PyEval_SaveThread();
}

static epicsThreadOnceId setupPyOnceId = EPICS_THREAD_ONCE_INIT;

void evalPy(const char* code)
{
    PyEval_RestoreThread(main_state);

    if(PyRun_SimpleStringFlags(code, NULL)!=0)
        PyErr_Print();

    main_state = PyEval_SaveThread();
}

void evalFilePy(const char* file)
{
    FILE *fp;

    PyEval_RestoreThread(main_state);

    fp = fopen(file, "r");
    if(!fp) {
        fprintf(stderr, "Failed to open: %s\n", file);
        perror("open");
    } else {
        if(PyRun_SimpleFileExFlags(fp, file, 1, NULL)!=0)
            PyErr_Print();
    }
    /* fp closed by python */

    main_state = PyEval_SaveThread();
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
}

#include <epicsExport.h>
epicsExportRegistrar(pySetupReg);
