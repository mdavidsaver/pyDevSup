
/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>

#include <epicsVersion.h>
#include <assert.h>
#include <dbCommon.h>
#include <dbStaticLib.h>
#include <dbAccess.h>
#include <devSup.h>
#include <dbLink.h>
#include <recGbl.h>
#include <alarm.h>
#include <ellLib.h>
#include <dbScan.h>
#include <cantProceed.h>

#include "pydevsup.h"

static ELLLIST devices = ELLLIST_INIT;

typedef struct {
    ELLNODE node;

    dbCommon *precord;

    DBLINK *plink;

    PyObject *pyrecord;
    PyObject *support;
    PyObject *scanobj;

    IOSCANPVT scan;

    PyObject *reason;
} pyDevice;

static long parse_link(dbCommon *prec, const char* src)
{
    PyObject *mod, *ret;
    pyDevice *priv = prec->dpvt;

    mod = PyImport_ImportModule("devsup.db");
    if(!mod)
        return -1;

    ret = PyObject_CallMethod(mod, "processLink", "ss", prec->name, src);
    Py_DECREF(mod);
    if(!ret)
        return -1;

    if(!PyArg_ParseTuple(ret, "OO", &priv->pyrecord, &priv->support)) {
        priv->pyrecord = priv->support = NULL; /* paranoia */
        Py_DECREF(ret);
        return -1;
    }

    Py_INCREF(priv->pyrecord);
    Py_INCREF(priv->support);
    Py_DECREF(ret);

    return 0;
}

static int allow_ioscan(pyDevice *priv)
{
    PyObject *ret = PyObject_CallMethod(priv->support, "allowScan", "O", priv->pyrecord);
    if(!ret) {
        PyErr_Print();
        PyErr_Clear();
    } else if(!PyObject_IsTrue(ret)) {
        Py_DECREF(ret);
    } else {
        priv->scanobj = ret;
        return 1;
    }
    return 0;
}

static void release_ioscan(pyDevice *priv)
{
    if(priv->scanobj && PyCallable_Check(priv->scanobj)) {
        PyObject *ret = PyObject_CallFunction(priv->scanobj, "O", priv->pyrecord);
        if(ret)
            Py_DECREF(ret);
        else {
            PyErr_Print();
            PyErr_Clear();
        }
    }
    Py_XDECREF(priv->scanobj);
    priv->scanobj = NULL;
}

static long detach_common(dbCommon *prec)
{
    pyDevice *priv = prec->dpvt;
    long ret = 0;

    release_ioscan(priv);

    if(priv->support) {
        PyObject *junk = 0, *sup=priv->support;
        junk = PyObject_CallMethod(sup, "detach", "O", priv->pyrecord);
        priv->support = NULL;
        Py_DECREF(sup);
        if(junk)
            Py_DECREF(junk);
        else
            ret = -1;
    }
    if(priv->pyrecord)
        Py_DECREF(priv->pyrecord);
    priv->pyrecord = NULL;

    return ret;
}

static long process_common(dbCommon *prec)
{
    pyDevice *priv = prec->dpvt;
    PyObject *reason = priv->reason;
    PyObject *ret;

    if(!reason)
        reason = Py_None;

    ret = PyObject_CallMethod(priv->support, "process", "OO", priv->pyrecord, reason);
    if(!ret)
        return -1;
    Py_DECREF(ret);
    return 0;
}

static long report(int lvl)
{
    PyGILState_STATE pystate;
    ELLNODE *cur;

    pystate = PyGILState_Ensure();
    if(lvl==0)
        printf("%d records\n", (int)ellCount(&devices));
    else
        for(cur=ellFirst(&devices); cur; cur=ellNext(cur))
        {
            pyDevice *priv=(pyDevice*)cur;
            printf("%s (%s)\n", priv->precord->name, priv->plink->value.instio.string);
            if(lvl>1) {
                PyObject *obj = priv->support ? priv->support : Py_None;
                printf("Support: ");
                PyObject_Print(obj, stdout, 0);

                obj = priv->scanobj ? priv->scanobj : Py_None;
                printf("IOSCAN: ");
                PyObject_Print(obj, stdout, 0);

                if(priv->reason) {
                    printf("Leftover reason!: ");
                    PyObject_Print(priv->reason, stdout, 0);
                }
            }
        }
    PyGILState_Release(pystate);
    return 0;
}

static long init_record(dbCommon *prec)
{
    return 0;
}

static long init_record2(dbCommon *prec)
{
    return 2;
}

static long add_record(dbCommon *prec)
{
    pyDevice *priv = prec->dpvt;
    PyGILState_STATE pystate;
    long ret;

    if(!priv) {
        DBENTRY ent;

        dbInitEntry(pdbbase, &ent);

        /* find ourself */
        ret = dbFindRecord(&ent, prec->name);
        assert(ret==0); /* really shouldn't fail */

        ret = dbFindField(&ent, "INP");

        if(ret)
            ret = dbFindField(&ent, "OUT");

        if(ret) {
            fprintf(stderr, "%s: Unable to find INP/OUT\n", prec->name);
            recGblSetSevr(prec, BAD_SUB_ALARM, INVALID_ALARM);
            return 0;
        }

        if(ent.pflddes->field_type!=DBF_INLINK
                && ent.pflddes->field_type!=DBF_OUTLINK)
        {
            fprintf(stderr, "%s: INP/OUT has unacceptible type %d\n",
                    prec->name, ent.pflddes->field_type);
            recGblSetSevr(prec, BAD_SUB_ALARM, INVALID_ALARM);
            return 0;
        }

        priv = callocMustSucceed(1, sizeof(*priv), "init_record");
        priv->precord = prec;
        priv->plink = ent.pfield;
        scanIoInit(&priv->scan);

        if(priv->plink->type != INST_IO) {
            fprintf(stderr, "%s: Has invalid link type %d\n", prec->name, priv->plink->type);
            recGblSetSevr(prec, BAD_SUB_ALARM, INVALID_ALARM);
            free(priv);
            return 0;
        }

        ellAdd(&devices, &priv->node);

        dbFinishEntry(&ent);
        prec->dpvt = priv;
    }

    assert(priv);
    assert(priv->plink->type == INST_IO);

    {
        char *msg=priv->plink->value.instio.string;
        if(!msg || *msg=='\0') {
            fprintf(stderr, "%s: Empty link string\n", prec->name);
            recGblSetSevr(prec, BAD_SUB_ALARM, INVALID_ALARM);
            return 0;
        }
    }

    pystate = PyGILState_Ensure();

    if(parse_link(prec, priv->plink->value.instio.string)) {
        fprintf(stderr, "%s: Exception in add_record\n", prec->name);
        PyErr_Print();
        PyErr_Clear();
        ret = S_db_errArg;
        goto done;
    }
    assert(priv->support);
    assert(priv->pyrecord);
    ret = 0;
done:
    PyGILState_Release(pystate);
    return ret;
}

static long del_record(dbCommon *prec)
{
    pyDevice *priv=prec->dpvt;
    PyGILState_STATE pystate;

    if(!priv || !priv->support)
        return 0;

    pystate = PyGILState_Ensure();

    if(detach_common(prec)) {
        fprintf(stderr, "%s: Exception in del_record\n", prec->name);
        PyErr_Print();
        PyErr_Clear();
    }

    assert(!priv->support);
    assert(!priv->pyrecord);

    PyGILState_Release(pystate);
    return 0;
}

static long get_iointr_info(int dir, dbCommon *prec, IOSCANPVT *scan)
{
    pyDevice *priv=prec->dpvt;
    PyGILState_STATE pystate;
    if(!priv || !priv->support)
        return 0;

    pystate = PyGILState_Ensure();

    if(dir==0) {
        if(!allow_ioscan(priv))
            return S_db_Blocked;
    } else {
        release_ioscan(priv);
    }

    PyGILState_Release(pystate);

    *scan = priv->scan;
    return 0;
}

static long process_record(dbCommon *prec)
{
    pyDevice *priv = prec->dpvt;
    PyGILState_STATE pystate;
    int pact = prec->pact;

    if(!priv || !priv->support)
        return 0;
    pystate = PyGILState_Ensure();

    if(process_common(prec)) {
        fprintf(stderr, "%s: Exception in process_record\n", prec->name);
        PyErr_Print();
        PyErr_Clear();
    }

    /* always clear PACT if it was initially set */
    if(pact)
        prec->pact = 0;

    PyGILState_Release(pystate);
    return 0;
}

int setReasonPyRecord(dbCommon *prec, PyObject *reason)
{
    pyDevice *priv=prec->dpvt;
    if(!isPyRecord(prec) || !priv || priv->reason)
        return 0;
    Py_INCREF(reason);
    priv->reason = reason;
    return 1;
}

int clearReasonPyRecord(dbCommon *prec)
{
    pyDevice *priv=prec->dpvt;
    if(!isPyRecord(prec) || !priv || !priv->reason)
        return 0;
    Py_DECREF(priv->reason);
    priv->reason = NULL;
    return 1;
}


static dsxt pydevsupExt = {&add_record, &del_record};


static long init(int i)
{
    if(i==0)
        devExtend(&pydevsupExt);
    return 0;
}


typedef struct {
    dset com;
    DEVSUPFUN proc;
} dset5;

static dset5 pydevsupCom = {{5, (DEVSUPFUN)&report, (DEVSUPFUN)&init,
                             (DEVSUPFUN)&init_record,
                             (DEVSUPFUN)&get_iointr_info},
                            (DEVSUPFUN)&process_record};
static dset5 pydevsupCom2 = {{5, (DEVSUPFUN)&report, (DEVSUPFUN)&init,
                              (DEVSUPFUN)&init_record2,
                              (DEVSUPFUN)&get_iointr_info},
                             (DEVSUPFUN)&process_record};

int isPyRecord(dbCommon *prec)
{
    return prec->dset==(dset*)&pydevsupCom || prec->dset==(dset*)&pydevsupCom2;
}

int canIOScanRecord(dbCommon *prec)
{
    pyDevice *priv=prec->dpvt;
    if(!isPyRecord(prec))
        return 0;
    return !!priv->scanobj;
}

/* Called with GIL locked */
void pyDBD_cleanup(void)
{
    ELLNODE *cur;
    while((cur=ellGet(&devices))!=NULL) {
        pyDevice *priv=(pyDevice*)cur;

        /* disconnect record by clearing DPVT */
        Py_BEGIN_ALLOW_THREADS {

            dbScanLock(priv->precord);
            assert(priv==priv->precord->dpvt);
            priv->precord->dpvt = NULL;
            dbScanUnlock(priv->precord);

        } Py_END_ALLOW_THREADS

        /* cleanup and dealloc */

        Py_XDECREF(priv->support);
        Py_XDECREF(priv->pyrecord);
        Py_XDECREF(priv->reason);
        priv->support = priv->pyrecord = priv->reason = NULL;

        free(priv);
    }
}

#include <epicsExport.h>

epicsExportAddress(dset, pydevsupCom);
epicsExportAddress(dset, pydevsupCom2);
