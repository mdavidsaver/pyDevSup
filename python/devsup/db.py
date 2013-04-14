
import threading, sys, traceback, time

from devsup.util import Worker

try:
    import _dbapi
except ImportError:
    import _nullapi as _dbapi

_rec_cache = {}
_no_such_field = object()

__all__ = [
    'Record', 'getRecord',
    'Field',
    'IOScanListBlock',
    'IOScanListThread',
]

def getRecord(name):
    """Retrieve a :class:`Record` instance by the
    full record name.
    
    The result is cached so the future calls will return the same instance.
    This is the prefered way to get :class:`Record` instances.
    
    >>> R = getRecord("my:record:name")
    Record("my:record:name")
    """
    try:
        return _rec_cache[name]
    except KeyError:
        rec = Record(name)
        _rec_cache[name] = rec
        return rec

class IOScanListBlock(object):
    """A list of records which will be processed together.
    
    This convienence class to handle the accounting to
    maintain a list of records.
    """
    def __init__(self):
        super(IOScanListBlock,self).__init__()
        self._recs, self._recs_add, self._recs_remove = set(), set(), set()
        self.force, self._running = 2, False

    def add(self, rec):
        """Add a record to the scan list.
        
        This method is designed to be consistent
        with :meth:`allowScan <DeviceSupport.allowScan>`
        by returning its :meth:`remove` method.
        If fact this function can be completely delegated. ::
        
          class MyDriver(util.StoppableThread):
            def __init__(self):
              super(MyDriver,self).__init__()
              self.lock = threading.Lock()
              self.scan1 = IOScanListBlock()
            def run(self):
              while self.shouldRun():
                time.sleep(1)
                with self.lock:
                  self.scan1.interrupt()
  
          class MySup(object):
            def __init__(self, driver):
              self.driver = driver
            def allowScan(rec):
              with self.driver.lock:
                return self.driver.scan1.add(rec)
        """
        assert isinstance(rec, Record)

        if self._running:
            self._recs_remove.discard(rec)
            self._recs_add.add(rec)

        else:
            self._recs.add(rec)

        return self.remove

    def remove(self, rec):
        """Remove a record from the scan list.
        """
        if self._running:
            self._recs_add.discard(rec)
            self._recs_add._recs_remove(rec)

        else:
            self._recs.discard(rec)

    def interrupt(self, reason=None, mask=None):
        """Scan the records in this list.

        :param reason: Passed to :meth:`Record.scan`.
        :param mask: A *list* or *set* or records which should not be scanned.

        This method will call :meth:`Record.scan` of each of the records
        currently in the list.  This is done synchronously in the current
        thread.  It should **never** be call when any record locks are held.
        """
        self._running = True
        try:
            for R in self._recs:
                if mask and R in mask:
                    continue
                R.scan(sync=True, reason=reason, force=self.force)
        finally:
            self._running = False
            if self._recs_add or self._recs_remove:
                assert len(self._recs_add.interaction(self._recs_remove))==0
            
                self._recs.update(self._recs_add)
                self._recs_add.clear()
                self._recs.difference_update(self._recs_remove)
                self._recs_remove.clear()
            

def _default_whendone(type, val, tb):
    if type or val or tb:
        traceback.print_exception(type, val, tb)

class IOScanListThread(IOScanListBlock):
    """A list of records w/ a worker thread to run them.
    
    All methods are thread-safe.
    """
    _worker = None
    _worker_lock = threading.Lock()
    queuelength=100
    @classmethod
    def getworker(cls):
        with cls._worker_lock:
            if cls._worker is not None:
                return cls._worker
            import devsup.hooks
            T = Worker(max=cls.queuelength)
            devsup.hooks.addHook('AtIocExit', T.join)
            T.start()
            cls._worker = T
            return T

    def __init__(self):
        super(IOScanListThread,self).__init__()
        self._lock = threading.Lock()

    def add(self, rec):
        """Add a record to the scan list.
        
        This method is thread-safe and may be used
        without additional locking. ::
        
          class MyDriver(util.StoppableThread):
            def __init__(self):
              super(MyDriver,self).__init__()
              self.scan1 = IOScanListThread()
            def run(self):
              while self.shouldRun():
                time.sleep(1)
                self.scan1.interrupt()
                
          class MySup(object):
            def __init__(self, driver):
              self.driver = driver
              self.allowScan = self.driver.scan1.add
        """
        with self._lock:
            return super(IOScanListThread,self).add(rec)

    def remove(self, rec):
        with self._lock:
            return super(IOScanListThread,self).remove(rec)

    def interrupt(self, reason=None, mask=None, whendone=_default_whendone):
        """Queue a request to process the scan list.

        :param reason: Passed to :meth:`Record.scan`.
        :param mask: A *list* or *set* or records which should not be scanned.
        :param whendone: A callable which will be invoked after all records are processed.
        :throws: RuntimeError is the request can't be queued.

        Calling this method will cause a request to be sent to a
        worker thread.  This method can be called several times
        to queue several requests.
        
        If provided, the *whendone* callable is invoked with three arguments.
        These will be None except in the case an interrupt is raised in the
        worker in which case they are: exception type, value, and traceback.
        
        .. note::
          This method may be safely called while record locks are held.
        """
        W = self.getworker()
        W.add(self._X, (reason, mask, whendone))

    def _X(self, reason, mask, whendone):
        try:
            #print 'XXXXX',self
            with self._lock:
                super(IOScanListThread,self).interrupt(reason, mask)
        finally:
            #print 'YYYYY',self,sys.exc_info()
            whendone(*sys.exc_info())

class Record(_dbapi._Record):
    def __init__(self, *args, **kws):
        super(Record, self).__init__(*args, **kws)
        super(Record, self).__setattr__('_fld_cache', {})

    def field(self, name):
        """Lookup field in this record
        
        :rtype: :class:`Field`
        :throws: KeyError for non-existant fields.
        
        The returned object is cached so future calls will
        return the same instance.
        
        >>> getRecord("rec").field('HOPR')
        Field("rec.HOPR")
        """
        try:
            F = self._fld_cache[name]
            if F is _no_such_field:
                raise ValueError()
            return F
        except KeyError as e:
            try:
                fld = Field("%s.%s"%(self.name(), name))
            except ValueError:
                self._fld_cache[name] = _no_such_field
                raise e
            else:
                self._fld_cache[name] = fld
                return fld

    def setTime(self, ts):
        """Set record timestamp.
        
        Has not effect if the TSE field is not set to -2.
        Accepts timetuple, float, or (sec, nsec).
        All inputs must be referenced to the posix epoch.
        """
        if hasattr(ts, 'timetuple'):
            ts = time.mktime(ts.timetuple())

        try:
            sec, nsec = ts
        except TypeError:
            sec = int(ts)
            nsec = int(ts*1e9)%1000000000

        super(Record, self).setTime(sec, nsec)

    def __getattr__(self, name):
        try:
            F = self.field(name)
        except ValueError:
            raise AttributeError('No such field')
        else:
            return F.getval()

    def __setattr__(self, name, val):
        try:
            F=self.field(name)
        except ValueError:
            super(Record, self).__setattr__(name, val)
        else:
            F.putval(val)


    def __repr__(self):
        return 'Record("%s")'%self.name()

class Field(_dbapi._Field):
    @property
    def record(self):
        """Fetch the :class:`Record` associated with this field
        """
        try:
            return self._record
        except AttributeError:
            rec, _ = self.name()
            self._record = getRecord(rec)
            return self._record

    def __cmp__(self, B):
        if isinstance(B, Field):
            B=B.getval()
        return cmp(self.getval(), B)

    def __int__(self):
        return int(self.getval())
    def __long__(self):
        return long(self.getval())
    def __float__(self):
        return float(self.getval())

    def __repr__(self):
        return 'Field("%s.%s")'%self.name()

def processLink(name, lstr):
    """Process the INP or OUT link
    
    Expects lstr to be "module arg1 arg2"

    Returns (callable, Record, "arg1 arg2")
    """
    rec = getRecord(name)
    parts = lstr.split(None,1)
    modname, args = parts[0], parts[1] if len(parts)>1 else None
    mod = __import__(modname)
    return rec, mod.build(rec, args)
