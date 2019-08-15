
/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>
#ifdef HAVE_NUMPY
#include <numpy/ndarrayobject.h>
#endif

#include <epicsVersion.h>
#include <dbCommon.h>
#include <dbAccess.h>
#include <dbStaticLib.h>
#include <recSup.h>
#include <special.h>

#include "pydevsup.h"

#if EPICS_VERSION>3 || (EPICS_VERSION==3 && EPICS_REVISION>=16)
#  define HAVE_INT64
#endif

#ifdef HAVE_NUMPY
static const int dbf2np_map[DBF_MENU+1] = {
    NPY_STRING,  // DBF_STRING
    NPY_BYTE,    // DBF_CHAR
    NPY_UBYTE,   // DBF_UCHAR
    NPY_INT16,   // DBF_SHORT
    NPY_UINT16,  // DBF_USHORT
    NPY_INT32,   // DBF_LONG
    NPY_UINT32,  // DBF_ULONG
#ifdef HAVE_INT64
    NPY_INT64,   // DBF_INT64
    NPY_UINT64,  // DBF_UINT64
#endif
    NPY_FLOAT32, // DBF_FLOAT
    NPY_FLOAT64, // DBF_DOUBLE
    NPY_INT16,   // DBF_ENUM
    NPY_INT16,   // DBF_MENU
};
static PyArray_Descr* dbf2np[DBF_MENU+1];
#endif

typedef struct {
    PyObject_HEAD

    DBADDR addr;
} pyField;

static int pyField_Init(pyField *self, PyObject *args, PyObject *kws)
{
    const char *pvname;

    if(!PyArg_ParseTuple(args, "s", &pvname))
        return -1;

    if(dbNameToAddr(pvname, &self->addr)) {
        PyErr_Format(PyExc_ValueError, "%s: Record field not found", pvname);
        return -1;
    }

    return 0;
}

static void exc_wrong_ftype(pyField *self)
{
    PyErr_Format(PyExc_TypeError, "Operation on field type %d is not supported",
                 self->addr.field_type);
}

static PyObject* pyField_name(pyField *self)
{
    return Py_BuildValue("ss",
                         self->addr.precord->name,
                         self->addr.pfldDes->name);
}

static PyObject* pyField_fldinfo(pyField *self)
{
    short dbf=self->addr.field_type;
    short fsize=self->addr.field_size;
    unsigned long nelm=self->addr.no_elements;
    return Py_BuildValue("hhk", dbf, fsize, nelm);
}

static PyObject* build_array(PyObject* obj, void *data, unsigned short ftype, unsigned long nelem, int flags)
{
#ifdef HAVE_NUMPY
    PyArray_Descr *desc;
    int ndims = 1;
    npy_intp dims[1] = {nelem};

    if(ftype>DBF_MENU) {
        PyErr_SetString(PyExc_TypeError, "Can not map field type to numpy type");
        return NULL;
    }

    desc = dbf2np[ftype];
    if(ftype==DBF_STRING) {
        desc->elsize = MAX_STRING_SIZE;
    }

    Py_XINCREF(desc);
    return PyArray_NewFromDescr(&PyArray_Type, desc, ndims, dims, NULL, data, flags, (PyObject*)obj);
#else
    PyErr_SetNone(PyExc_NotImplementedError);
    return NULL;
#endif
}

static int assign_array(DBADDR *paddr, PyObject *arr)
{
#ifdef HAVE_NUMPY
    void *rawfield = paddr->pfield;
    rset *prset;
    PyObject *aval;
    unsigned elemsize = dbValueSize(paddr->field_type);
    unsigned long maxlen = paddr->no_elements, insize;
    PyArray_Descr *desc = dbf2np[paddr->field_type];

    if(paddr->field_type==DBF_STRING &&
        (PyArray_NDIM(arr)!=2 || PyArray_DIM(arr,0)>maxlen || PyArray_DIM(arr,1)!=MAX_STRING_SIZE))
    {
        PyErr_Format(PyExc_ValueError, "String array has incorrect shape or is too large");
        return 1;

    } else if(PyArray_NDIM(arr)!=1 || PyArray_DIM(arr,0)>maxlen) {
        PyErr_Format(PyExc_ValueError, "Array has incorrect shape or is too large");
        return 1;
    }

    insize = PyArray_DIM(arr, 0);

    if(paddr->special==SPC_DBADDR &&
       (prset=dbGetRset(paddr)) &&
       prset->get_array_info)
    {
        /* array */
        char *datasave=paddr->pfield;
        long noe, off;
        if(prset->get_array_info(paddr, &noe, &off)) {
            PyErr_Format(PyExc_ValueError, "Error fetching array info for %s.%s",
                     paddr->precord->name,
                     paddr->pfldDes->name);
            return 1;
        }

        rawfield = paddr->pfield;
        /* get_array_info can modify pfield in >3.15.0.1 */
        paddr->pfield = datasave;
    }

    Py_XINCREF(desc);
    if(!(aval = PyArray_FromAny(arr, desc, 1, 2, NPY_CARRAY, arr)))
        return 1;

    if(elemsize!=PyArray_ITEMSIZE(aval)) {
        PyErr_Format(PyExc_AssertionError, "item size mismatch %u %u",
                    elemsize, (unsigned)PyArray_ITEMSIZE(aval) );
        return 1;
    }

    memcpy(rawfield, PyArray_GETPTR1(aval, 0), insize*elemsize);

    Py_DECREF(aval);

    if(paddr->special==SPC_DBADDR &&
       (prset=dbGetRset(paddr)) &&
       prset->get_array_info)
    {
        if(prset->put_array_info(paddr, insize)) {
            PyErr_Format(PyExc_ValueError, "Error setting array info for %s.%s",
                         paddr->precord->name,
                         paddr->pfldDes->name);
            return 1;
        }
    }

    return 0;
#else
    PyErr_SetNone(PyExc_NotImplementedError);
    return 1;
#endif
}

static PyObject* pyField_getval(pyField *self)
{
    void *rawfield = self->addr.pfield;
    rset *prset;

    if(self->addr.special==SPC_DBADDR &&
       (prset=dbGetRset(&self->addr)) &&
       prset->get_array_info)
    {
        /* array */
        char *datasave=self->addr.pfield;
        long noe, off;
        if(prset->get_array_info(&self->addr, &noe, &off))
            return PyErr_Format(PyExc_ValueError, "Error fetching array info for %s.%s",
                     self->addr.precord->name,
                     self->addr.pfldDes->name);

        rawfield = self->addr.pfield;
        /* get_array_info can modify pfield in >3.15.0.1 */
        self->addr.pfield = datasave;

        if(self->addr.no_elements>1) {
            return build_array((PyObject*)self, rawfield, self->addr.field_type,
                               noe, NPY_CARRAY_RO);
        }
    }

    switch(self->addr.field_type)
    {
#define OP(FTYPE, CTYPE, FN) case DBF_##FTYPE: return FN(*(CTYPE*)rawfield)
    OP(CHAR,  epicsInt8,   PyInt_FromLong);
    OP(UCHAR, epicsUInt8,  PyInt_FromLong);
    OP(ENUM,  epicsEnum16, PyInt_FromLong);
    OP(MENU,  epicsEnum16, PyInt_FromLong);
    OP(SHORT, epicsInt16,  PyInt_FromLong);
    OP(USHORT,epicsUInt16, PyInt_FromLong);
    OP(LONG,  epicsInt32,  PyInt_FromLong);
    OP(ULONG, epicsUInt32, PyInt_FromLong);
#ifdef HAVE_INT64
    OP(INT64,  epicsInt64,  PyLong_FromLongLong);
    OP(UINT64, epicsUInt64, PyLong_FromLongLong);
#endif
    OP(FLOAT, epicsFloat32,PyFloat_FromDouble);
    OP(DOUBLE,epicsFloat64,PyFloat_FromDouble);
#undef OP
    case DBF_STRING:
        return PyString_FromString((char*)rawfield);
    default:
        exc_wrong_ftype(self);
        return NULL;
    }
}

static PyObject* pyField_putval(pyField *self, PyObject* args)
{
    PyObject *val;
    void *rawfield;
    rset *prset;

    if(!PyArg_ParseTuple(args, "O", &val))
        return NULL;

    if(self->addr.field_type>DBF_MENU) {
        PyErr_SetString(PyExc_TypeError, "field type write not supported");
        return NULL;
    }

    if(val==Py_None) {
        PyErr_Format(PyExc_ValueError, "Can't assign None to %s.%s",
                     self->addr.precord->name,
                     self->addr.pfldDes->name);
        return NULL;
    }

#ifdef HAVE_NUMPY
    if(PyArray_Check(val)) { /* assign from array */
        if(assign_array(&self->addr, val))
            return NULL;
        Py_RETURN_NONE;
    }
#endif

    if(self->addr.special==SPC_DBADDR &&
       (prset=dbGetRset(&self->addr)) &&
       prset->get_array_info)
    {
        /* writing scalar to array */
        char *datasave=self->addr.pfield;
        long noe, off; /* ignored */
        if(prset->get_array_info(&self->addr, &noe, &off))
            return PyErr_Format(PyExc_ValueError, "Error fetching array info for %s.%s",
                     self->addr.precord->name,
                     self->addr.pfldDes->name);
        rawfield = self->addr.pfield;
        /* get_array_info can modify pfield in >3.15.0.1 */
        self->addr.pfield = datasave;

    } else
        rawfield = self->addr.pfield;

    switch(self->addr.field_type)
    {
#define OP(FTYPE, CTYPE, FN) case DBF_##FTYPE: *(CTYPE*)rawfield = FN(val); break
    OP(CHAR,  epicsInt8,   PyInt_AsLong);
    OP(UCHAR, epicsUInt8,  PyInt_AsLong);
    OP(ENUM,  epicsEnum16, PyInt_AsLong);
    OP(MENU,  epicsEnum16, PyInt_AsLong);
    OP(SHORT, epicsInt16,  PyInt_AsLong);
    OP(USHORT,epicsUInt16, PyInt_AsLong);
    OP(LONG,  epicsInt32,  PyInt_AsLong);
    OP(ULONG, epicsUInt32, PyInt_AsLong);
#ifdef HAVE_INT64
    OP(INT64,  epicsInt32,  PyLong_AsLongLong);
    OP(UINT64, epicsUInt32, PyLong_AsLongLong);
#endif
    OP(FLOAT, epicsFloat32,PyFloat_AsDouble);
    OP(DOUBLE,epicsFloat64,PyFloat_AsDouble);
#undef OP
    case DBF_STRING: {
        const char *fld;
        char *dest=rawfield;
#if PY_MAJOR_VERSION >= 3
        PyObject *data = PyUnicode_AsASCIIString(val);
        if(!data)
            return NULL;
        fld = PyBytes_AS_STRING(data);
#else
        fld = PyString_AsString(val);
#endif
        if(fld) {
          strncpy(dest, fld, MAX_STRING_SIZE);
          dest[MAX_STRING_SIZE-1]='\0';
        } else {
          dest[0] = '\0';
        }
#if PY_MAJOR_VERSION >= 3
        Py_DECREF(data);
#endif
        break;
    }
    default:
        exc_wrong_ftype(self);
        return NULL;
    }


    if(self->addr.special==SPC_DBADDR &&
       (prset=dbGetRset(&self->addr)) &&
       prset->get_array_info)
    {
        if(prset->put_array_info(&self->addr, 1))
            return PyErr_Format(PyExc_ValueError, "Error setting array info for %s.%s",
                     self->addr.precord->name,
                     self->addr.pfldDes->name);
    }

    Py_RETURN_NONE;
}

static PyObject *pyField_getarray(pyField *self)
{
    rset *prset;
    char *data;

    if(self->addr.special==SPC_DBADDR &&
       (prset=dbGetRset(&self->addr)) &&
       prset->get_array_info)
    {
        char *datasave=self->addr.pfield;
        long noe, off; /* ignored */
        prset->get_array_info(&self->addr, &noe, &off);
        data = self->addr.pfield;
        /* get_array_info can modify pfield in >3.15.0.1 */
        self->addr.pfield = datasave;

    } else
        data = self->addr.pfield;

    return build_array((PyObject*)self, data, self->addr.field_type, self->addr.no_elements, NPY_CARRAY);
}

static PyObject *pyField_getlen(pyField *self)
{
    rset *prset;

    if(self->addr.special==SPC_DBADDR &&
       (prset=dbGetRset(&self->addr)) &&
       prset->get_array_info)
    {
        char *datasave=self->addr.pfield;
        long noe, off;
        prset->get_array_info(&self->addr, &noe, &off);
        /* get_array_info can modify pfield in >3.15.0.1 */
        self->addr.pfield = datasave;
        return PyInt_FromLong(noe);

    } else
        return PyInt_FromLong(1);
}

static PyObject *pyField_setlen(pyField *self, PyObject *args)
{
    rset *prset = dbGetRset(&self->addr);
    Py_ssize_t len;

    if(!PyArg_ParseTuple(args, "n", &len))
        return NULL;

    if(self->addr.special!=SPC_DBADDR ||
       !prset ||
       !prset->put_array_info)
    {
        PyErr_SetString(PyExc_TypeError, "Not an array field");
        return NULL;
    }

    if(len > self->addr.no_elements) {
        PyErr_Format(PyExc_ValueError, "Requested length %ld out of range [0,%lu)",
                        (long)len, (unsigned long)self->addr.no_elements);
        return NULL;
    }

    prset->put_array_info(&self->addr, len);

    Py_RETURN_NONE;
}

static PyObject* pyField_getTime(pyField *self)
{
    DBLINK *plink = self->addr.pfield;
    epicsTimeStamp ret;
    long sts, sec, nsec;

    if(self->addr.field_type!=DBF_INLINK) {
        PyErr_SetString(PyExc_TypeError, "getTime can only be used on DBF_INLINK fields");
        return NULL;
    }

    sts = dbGetTimeStamp(plink, &ret);
    if(sts) {
        PyErr_SetString(PyExc_TypeError, "getTime failed");
        return NULL;
    }

    sec = ret.secPastEpoch + POSIX_TIME_AT_EPICS_EPOCH;
    nsec = ret.nsec;

    return Py_BuildValue("ll", sec, nsec);
}

static PyObject* pyField_getAlarm(pyField *self)
{
    DBLINK *plink = self->addr.pfield;
    long sts;
    epicsEnum16 stat, sevr;

    if(self->addr.field_type!=DBF_INLINK) {
        PyErr_SetString(PyExc_TypeError, "getAlarm can only be used on DBF_INLINK fields");
        return NULL;
    }

    sts = dbGetAlarm(plink, &stat, &sevr);
    if(sts) {
        PyErr_SetString(PyExc_TypeError, "getAlarm failed");
        return NULL;
    }

    return Py_BuildValue("ll", (long)sevr, (long)stat);
}

static PyObject *pyField_len(pyField *self)
{
    return PyInt_FromLong(self->addr.no_elements);
}

static PyMethodDef pyField_methods[] = {
    {"name", (PyCFunction)pyField_name, METH_NOARGS,
     "name() -> (recname, fldname)\n"},
    {"fieldinfo", (PyCFunction)pyField_fldinfo, METH_NOARGS,
     "fieldinfo() -> (dbf, elem_size, elem_count"},
    {"getval", (PyCFunction)pyField_getval, METH_NOARGS,
     "getval() -> object\n"},
    {"putval", (PyCFunction)pyField_putval, METH_VARARGS,
     "putval(object)\n"},
    {"getarray", (PyCFunction)pyField_getarray, METH_NOARGS,
     "getarray() -> numpy.ndarray\n"
     "Return a numpy ndarray refering to this field for in-place operations."},
    {"getarraylen", (PyCFunction)pyField_getlen, METH_NOARGS,
     "getarraylen() -> int\n"
     "Return current number of valid elements for array fields."},
    {"putarraylen", (PyCFunction)pyField_setlen, METH_VARARGS,
     "putarraylen(int)\n"
     "Set number of valid elements for array fields."},
    {"getTime", (PyCFunction)pyField_getTime, METH_NOARGS,
     "getTime() -> (sec, nsec)."},
    {"getAlarm", (PyCFunction)pyField_getAlarm, METH_NOARGS,
     "getAlarm() -> (severity, status)."},
    {"__len__", (PyCFunction)pyField_len, METH_NOARGS,
     "Maximum number of elements storable in this field."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION < 3
static Py_ssize_t pyField_buf_getcount(pyField *self, Py_ssize_t *totallen)
{
    if(totallen)
        *totallen = self->addr.field_size * self->addr.no_elements;
    return 1;
}

static Py_ssize_t pyField_buf_getbuf(pyField *self, Py_ssize_t bufid, void **data)
{
    if(bufid!=0) {
        PyErr_SetString(PyExc_SystemError, "Requested invalid segment");
        return -1;
    }
    *data = self->addr.pfield;
    return self->addr.field_size * self->addr.no_elements;
}

static Py_ssize_t pyField_buf_getcharbuf(pyField *self, Py_ssize_t bufid, char **data)
{
    if(bufid!=0) {
        PyErr_SetString(PyExc_SystemError, "Requested invalid segment");
        return -1;
    } else if(self->addr.field_size!=1) {
        PyErr_SetString(PyExc_TypeError, "Field type must be CHAR or UCHAR");
        return -1;
    }
    *data = self->addr.pfield;
    return self->addr.field_size * self->addr.no_elements;
}
#endif

static int pyField_buf_getbufferproc(pyField *self, Py_buffer *buf, int flags)
{
    return PyBuffer_FillInfo(buf, (PyObject*)self,
                             self->addr.pfield,
                             self->addr.field_size * self->addr.no_elements,
                             0, flags);
}

static PyBufferProcs pyField_buf_methods = {
#if PY_MAJOR_VERSION < 3
    (readbufferproc)pyField_buf_getbuf,
    (writebufferproc)pyField_buf_getbuf,
    (segcountproc)pyField_buf_getcount,
    (charbufferproc)pyField_buf_getcharbuf,
#endif
    (getbufferproc)pyField_buf_getbufferproc,
    (releasebufferproc)NULL,
};


static PyTypeObject pyField_type = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,
#endif
    "_dbapi._Field",
    sizeof(pyField),
};

int pyField_prepare(PyObject *module)
{
    size_t i;

    import_array1(-1);

    pyField_type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
#if PY_MAJOR_VERSION < 3
    pyField_type.tp_flags |= Py_TPFLAGS_HAVE_GETCHARBUFFER|Py_TPFLAGS_HAVE_NEWBUFFER;
#endif
    pyField_type.tp_methods = pyField_methods;
    pyField_type.tp_as_buffer = &pyField_buf_methods;
    pyField_type.tp_init = (initproc)pyField_Init;

    pyField_type.tp_new = PyType_GenericNew;
    if(PyType_Ready(&pyField_type)<0)
        return -1;

    PyObject *typeobj=(PyObject*)&pyField_type;
    Py_INCREF(typeobj);
    if(PyModule_AddObject(module, "_Field", typeobj)) {
        Py_DECREF(typeobj);
        return -1;
    }

#ifdef HAVE_NUMPY
    for(i=0; i<=DBF_MENU; i++) {
        dbf2np[i] = PyArray_DescrFromType(dbf2np_map[i]);
        assert(dbf2np[i]);
    }
#endif

    return 0;
}

