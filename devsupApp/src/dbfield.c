
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

#include "pydevsup.h"

#ifdef HAVE_NUMPY
static int dbf2np_map[DBF_ENUM+1] = {
    NPY_BYTE,
    NPY_BYTE,
    NPY_UBYTE,
    NPY_INT16,
    NPY_UINT16,
    NPY_INT32,
    NPY_UINT32,
    NPY_FLOAT32,
    NPY_FLOAT64,
    NPY_INT16,
};
static PyArray_Descr* dbf2np[DBF_ENUM+1];
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
        PyErr_SetString(PyExc_ValueError, "Record field not found");
        return -1;
    }

    if(self->addr.field_type >= DBF_ENUM) {
        PyErr_SetString(PyExc_ValueError, "Access to this field type is not supported");
        return -1;
    }
    return 0;
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

union dbfvalue {
    char dv_sval[MAX_STRING_SIZE];
    epicsUInt32 dv_uval;
    epicsInt32 dv_ival;
    double dv_fval;
};

static PyObject* pyField_getval(pyField *self)
{
    union dbfvalue buf;
    long nReq = 1;
    short reqtype;

    switch(self->addr.field_type)
    {
    case DBF_CHAR:
    case DBF_UCHAR:
    case DBF_SHORT:
    case DBF_USHORT:
    case DBF_ENUM:
    case DBF_LONG:
        reqtype = DBF_LONG;
        break;
    case DBF_ULONG:
        reqtype = DBF_ULONG;
        break;
    case DBF_FLOAT:
    case DBF_DOUBLE:
        reqtype = DBF_DOUBLE;
        break;
    case DBF_STRING:
        reqtype = DBF_STRING;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Access to this field type is not supported");
        return NULL;
    }

    if(dbGet(&self->addr, reqtype, buf.dv_sval, NULL, &nReq, NULL)) {
        PyErr_SetString(PyExc_ValueError, "Error getting field value");
        return NULL;
    }

    switch(reqtype) {
    case DBF_LONG:
        return PyInt_FromLong(buf.dv_ival);
    case DBF_ULONG:
        return PyInt_FromLong(buf.dv_uval);
    case DBF_DOUBLE:
        return PyInt_FromLong(buf.dv_fval);
    case DBF_STRING:
        return PyString_FromString(buf.dv_sval);
    default:
        Py_RETURN_NONE;
    }
}


static PyObject* pyField_putval(pyField *self, PyObject* args)
{
    PyObject *val;

    union dbfvalue buf;
    short puttype;

    if(!PyArg_ParseTuple(args, "O", &val))
        return NULL;

    if(PyFloat_Check(val)) {
        double v = PyFloat_AsDouble(val);
        buf.dv_fval = v;
        puttype = DBF_DOUBLE;

    } else if(PyInt_Check(val)) {
        long v = PyInt_AsLong(val);
        if(v>0x7fffffffL) {
            buf.dv_uval = v;
            puttype = DBF_ULONG;
        } else {
            buf.dv_ival = v;
            puttype = DBF_LONG;
        }

    } else if(PyString_Check(val)) {
        const char *v = PyString_AsString(val);
        strncpy(buf.dv_sval, v, MAX_STRING_SIZE);
        buf.dv_sval[MAX_STRING_SIZE-1] = '\0';
        puttype = DBF_STRING;

    } else {
        PyErr_SetString(PyExc_TypeError, "Unable to convert put type");
        return NULL;
    }

    if(dbPut(&self->addr, puttype, buf.dv_sval, 1)){
        PyErr_SetString(PyExc_ValueError, "Error putting field value");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *pyField_getarray(pyField *self)
{
#ifdef HAVE_NUMPY
    int flags = NPY_CARRAY;
    char *data=self->addr.pfield;
    npy_int dims[1] = {self->addr.no_elements};
    PyArray_Descr *desc;

    if(self->addr.field_type>DBF_ENUM) {
        PyErr_SetString(PyExc_TypeError, "Can not map field type to numpy type");
        return NULL;
    } else if(self->addr.field_type==DBF_STRING)
        dims[0] *= self->addr.field_size;

    desc = dbf2np[self->addr.field_type];
    Py_XINCREF(desc);
    return PyArray_NewFromDescr(&PyArray_Type, desc, 1, dims, NULL, data, flags, (PyObject*)self);

#else
    PyErr_SetNone(PyExc_NotImplementedError);
    return NULL;
#endif
}

static PyMethodDef pyField_methods[] = {
    {"name", (PyCFunction)pyField_name, METH_NOARGS,
     "Return Names (\"record\",\"field\")"},
    {"fieldinfo", (PyCFunction)pyField_fldinfo, METH_NOARGS,
     "Field type info\nReturn (type, size, #elements"},
    {"getval", (PyCFunction)pyField_getval, METH_NOARGS,
     "Returns scalar version of field value"},
    {"putval", (PyCFunction)pyField_putval, METH_VARARGS,
     "Sets field value from a scalar"},
    {"getarray", (PyCFunction)pyField_getarray, METH_NOARGS,
     "Return a numpy ndarray refering to this field for in-place operations."},
    {NULL, NULL, 0, NULL}
};

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
    return 0;
}

static Py_ssize_t pyField_buf_getcharbuf(pyField *self, Py_ssize_t bufid, char **data)
{
    if(bufid!=0) {
        PyErr_SetString(PyExc_SystemError, "Requested invalid segment");
        return -1;
    }
    if(self->addr.field_size!=1) {
        PyErr_SetString(PyExc_TypeError, "Field type must be CHAR or UCHAR");
        return -1;
    }
    *data = self->addr.pfield;
    return 0;
}

static int pyField_buf_getbufferproc(pyField *self, Py_buffer *buf, int flags)
{
    return PyBuffer_FillInfo(buf, (PyObject*)self,
                             self->addr.pfield,
                             self->addr.field_size * self->addr.no_elements,
                             0, flags);
}

static PyBufferProcs pyField_buf_methods = {
    (readbufferproc)pyField_buf_getbuf,
    (writebufferproc)pyField_buf_getbuf,
    (segcountproc)pyField_buf_getcount,
    (charbufferproc)pyField_buf_getcharbuf,
    (getbufferproc)pyField_buf_getbufferproc,
    (releasebufferproc)NULL,
};


static PyTypeObject pyField_type = {
    PyObject_HEAD_INIT(NULL)
    0,
    "_dbapi._Field",
    sizeof(pyField),
};

int pyField_prepare(void)
{
    size_t i;

    pyField_type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    pyField_type.tp_flags |= Py_TPFLAGS_HAVE_GETCHARBUFFER|Py_TPFLAGS_HAVE_NEWBUFFER;
    pyField_type.tp_methods = pyField_methods;
    pyField_type.tp_as_buffer = &pyField_buf_methods;
    pyField_type.tp_init = (initproc)pyField_Init;

    pyField_type.tp_new = PyType_GenericNew;
    if(PyType_Ready(&pyField_type)<0)
        return -1;

    import_array1(-1);

#ifdef HAVE_NUMPY
    for(i=0; i<=DBF_ENUM; i++) {
        dbf2np[i] = PyArray_DescrFromType(dbf2np_map[i]);
        assert(dbf2np[i]);
    }
#endif

    return 0;
}

void pyField_setup(PyObject *module)
{
    PyObject *typeobj=(PyObject*)&pyField_type;
    Py_INCREF(typeobj);
    PyModule_AddObject(module, "_Field", (PyObject*)&pyField_type);
}

void pyField_cleanup(void)
{
    size_t i;

    for(i=0; i<=DBF_ENUM; i++) {
        Py_XDECREF(dbf2np[i]);
        dbf2np[i] = NULL;
    }
}
