
import threading, sys, traceback

from util import Worker

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
    try:
        return _rec_cache[name]
    except KeyError:
        rec = Record(name)
        _rec_cache[name] = rec
        return rec

class IOScanListBlock(object):
    def __init__(self):
        super(IOScanListBlock,self).__init__()
        self._recs, self._recs_add, self._recs_remove = set(), set(), set()
        self.force, self._running = 2, False

    def add(self, rec):
        assert isinstance(rec, Record)

        if self._running:
            self._recs_remove.discard(rec)
            self._recs_add.add(rec)

        else:
            self._recs.add(rec)

        return self.remove

    def remove(self, rec):
        if self._running:
            self._recs_add.discard(rec)
            self._recs_add._recs_remove(rec)

        else:
            self._recs.discard(rec)

    def interrupt(self, reason=None, mask=None):
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
    _worker = None
    _worker_lock = threading.Lock()
    queuelength=100
    @classmethod
    def getworker(cls):
        with cls._worker_lock:
            if cls._worker is not None:
                return cls._worker
            import hooks
            T = Worker(max=cls.queuelength)
            hooks.addHook('AtIocExit', T.join)
            T.start()
            cls._worker = T
            return T

    def __init__(self):
        super(IOScanListThread,self).__init__()
        self._lock = threading.Lock()

    def add(self, rec):
        with self._lock:
            return super(IOScanListThread,self).add(rec)

    def remove(self, rec):
        with self._lock:
            return super(IOScanListThread,self).remove(rec)

    def interrupt(self, reason=None, mask=None, whendone=_default_whendone):
        W = self.getworker()
        try:
            W.add(self._X, (reason, mask, whendone))
            return True
        except RuntimeError:
            return False

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
        
        fld = rec.field('HOPR')
        """
        try:
            F = self._fld_cache[name]
            if F is _no_such_field:
                raise ValueError()
            return F
        except KeyError:
            try:
                fld = Field("%s.%s"%(self.name(), name))
            except ValueError:
                self._fld_cache[name] = _no_such_field
            else:
                self._fld_cache[name] = fld
                return fld

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
        """Fetch the record associated with this field
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
