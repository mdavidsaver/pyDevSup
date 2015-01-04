
#include <sys/inotify.h>
#include <unistd.h>
#include <fcntl.h>

/* python has its own ideas about which version to support */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <Python.h>
#include <structmember.h>

#if PY_MAJOR_VERSION >= 3
# define PyInt_FromLong PyLong_FromLong
#endif

#define EVTMAXSIZE (sizeof(struct inotify_event) + NAME_MAX + 1)
#define EVTMINSIZE offsetof(struct inotify_event,name)

typedef struct {
    PyObject_HEAD

    int fd;
    char buf[EVTMAXSIZE*16];
} INotify;

static int INotify_Init(INotify *self, PyObject *args, PyObject *kws)
{
    int flags;
    self->fd = inotify_init();
    if(self->fd==-1) {
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    flags = fcntl(self->fd, F_GETFL, 0);
    flags |= O_NONBLOCK;
    if(fcntl(self->fd, F_SETFL, flags)) {
        close(self->fd);
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    return 0;
}

static void INotify_dealloc(INotify *self)
{
    close(self->fd);
    Py_TYPE(self)->tp_free(self);
}

static PyObject* INotify_add_watch(INotify* self, PyObject* args)
{
    int ret;
    const char* path;
    unsigned long mask;

    if(!PyArg_ParseTuple(args, "sk", &path, &mask))
        return NULL;

    ret = inotify_add_watch(self->fd, path, mask);
    if(ret==-1) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    return PyInt_FromLong(ret);
}

static PyObject* INotify_rm_watch(INotify* self, PyObject* args)
{
    int wd, ret;

    if(!PyArg_ParseTuple(args, "i", &wd))
        return NULL;

    ret = inotify_rm_watch(self->fd, wd);
    if(ret==-1) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    Py_RETURN_NONE;
}

/* Reading inotify events (circa. Linux 3.12)
 * Based on a reading of copy_event_to_user() in inotify_user.c
 *
 * The read() call must be given a buffer big enough for at least one
 * event.  As event size is variable this means the buffer must
 * be sized for the woust cast (sizeof(inotify_event)+NAME_MAX+1).
 * We will only be given complete events, but can be sure how many.
 *
 * If we don't allocate enough space for one event read() gives EINVAL.
 * If we don't allocate enough space for two events, read() gives
 * the size of the first event.
 * read() should never return zero.
 */

static PyObject* INotify_read(INotify* self)
{
    PyObject *list = NULL;
    void *buf = self->buf;
    ssize_t ret;

    list = PyList_New(0);
    if(!list)
        return NULL;

retry:
    ret = read(self->fd, buf, sizeof(self->buf));
    if(ret<0) {
        if(errno==EAGAIN)
            return list; /* return empty list */
        else if(errno==EINTR) {
            if(PyErr_CheckSignals()==0)
                goto retry;
        }
        PyErr_SetFromErrno(PyExc_OSError);
        goto fail;
    } else if(ret<EVTMINSIZE) {
        PyErr_Format(PyExc_OSError, "The unthinkable has happened in INotify_read");
        goto fail;
    }

    while(ret>=EVTMINSIZE) {
        PyObject *tuple;
        struct inotify_event *evt=buf;
        ssize_t evtsize;

        /* paranoia validation */
        if(evt->len > ret) {
            PyErr_Format(PyExc_OSError, "Recieved event length %lu beyond buffer size %lu",
                         (unsigned long)evt->len, (unsigned long)ret);
            /* oops, we can't recover from this...  */
            close(self->fd);
            self->fd = -1;
            goto fail;
        } else if(evt->len>0)
            evt->name[evt->len-1] = '\0';
        else
            evt->name[0] = '\0';

        evtsize = (void*)&evt->name[evt->len] - buf;

        tuple = Py_BuildValue("iIIs",
                              (int)evt->wd, (unsigned int)evt->mask,
                              (unsigned int)evt->cookie,
                              evt->name);
        if(!tuple)
            goto fail;

        if(PyList_Append(list, tuple)) {
            Py_DECREF(tuple);
            goto fail;
        }
        Py_DECREF(tuple); /* PyList_Append() takes a reference */

        buf += evtsize;
        ret -= evtsize;
    }

    if(ret!=0)
        PyErr_Warn(PyExc_UserWarning, "Stray bytes in INotify_read");

    return list;
fail:
    Py_XDECREF(list);
    return NULL;
}

static struct PyMemberDef INotify_members[] = {
    {"fd", T_INT, offsetof(INotify, fd), READONLY,
     "Underlying file descriptor for notifications"},
    {NULL}
};

static struct PyMethodDef INotify_methods[] = {
    {"add", (PyCFunction)INotify_add_watch, METH_VARARGS,
     "Add a new path to watch"},
    {"_del", (PyCFunction)INotify_rm_watch, METH_VARARGS,
     "Stop watching a path"},
    {"read", (PyCFunction)INotify_read, METH_NOARGS,
     "Read one event"},
    {NULL}
};

static PyTypeObject INotify_type = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,
#endif
    "_inotifyy.INotify",
    sizeof(INotify),
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef inotifymodule = {
  PyModuleDef_HEAD_INIT,
    "_inotify",
    NULL,
    -1,
    NULL
};
#endif


#if PY_MAJOR_VERSION >= 3
# define MODINIT_RET(VAL) return (VAL)
#else
# define MODINIT_RET(VAL) return
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__inotifyy(void)
#else
PyMODINIT_FUNC init_inotifyy(void)
#endif
{
    PyObject *mod = NULL;

#if PY_MAJOR_VERSION >= 3
    mod = PyModule_Create(&inotifymodule);
#else
    mod = Py_InitModule("_inotifyy", NULL);
#endif
    if(!mod)
        goto fail;

    INotify_type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    INotify_type.tp_members = INotify_members;
    INotify_type.tp_methods = INotify_methods;
    INotify_type.tp_init = (initproc)INotify_Init;
    INotify_type.tp_dealloc = (destructor)INotify_dealloc;

    INotify_type.tp_new = PyType_GenericNew;
    if(PyType_Ready(&INotify_type)<0) {
        fprintf(stderr, "INotify object not ready\n");
        MODINIT_RET(NULL);
    }

    PyObject *typeobj=(PyObject*)&INotify_type;
    Py_INCREF(typeobj);
    if(PyModule_AddObject(mod, "INotify", typeobj)) {
        Py_DECREF(typeobj);
        fprintf(stderr, "Failed to add INotify object to module\n");
        MODINIT_RET(NULL);
    }

    PyModule_AddIntMacro(mod, IN_ACCESS);
    PyModule_AddIntMacro(mod, IN_ATTRIB);
    PyModule_AddIntMacro(mod, IN_CLOSE_WRITE);
    PyModule_AddIntMacro(mod, IN_CLOSE_NOWRITE);
    PyModule_AddIntMacro(mod, IN_CREATE);
    PyModule_AddIntMacro(mod, IN_DELETE);
    PyModule_AddIntMacro(mod, IN_DELETE_SELF);
    PyModule_AddIntMacro(mod, IN_MODIFY);
    PyModule_AddIntMacro(mod, IN_MOVE_SELF);
    PyModule_AddIntMacro(mod, IN_MOVED_FROM);
    PyModule_AddIntMacro(mod, IN_MOVED_TO);
    PyModule_AddIntMacro(mod, IN_OPEN);

    PyModule_AddIntMacro(mod, IN_ALL_EVENTS);

    PyModule_AddIntMacro(mod, IN_ONESHOT);
    /* added in glibc 2.5 */
#ifdef IN_EXCL_UNLINK
    PyModule_AddIntMacro(mod, IN_EXCL_UNLINK);
#endif
#ifdef IN_DONT_FOLLOW
    PyModule_AddIntMacro(mod, IN_DONT_FOLLOW);
#endif
#ifdef IN_MASK_ADD
    PyModule_AddIntMacro(mod, IN_MASK_ADD);
#endif
#ifdef IN_ONLYDIR
    PyModule_AddIntMacro(mod, IN_ONLYDIR);
#endif

    PyModule_AddIntMacro(mod, IN_IGNORED);
    PyModule_AddIntMacro(mod, IN_ISDIR);
    PyModule_AddIntMacro(mod, IN_Q_OVERFLOW);
    PyModule_AddIntMacro(mod, IN_UNMOUNT);

    MODINIT_RET(mod);

fail:
    fprintf(stderr, "Failed to initialize _inotify module!\n");
    Py_XDECREF(mod);
    MODINIT_RET(NULL);
}
