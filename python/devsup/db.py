try:
    import _dbapi
except ImportError:
    import _nullapi as _dbapi
from _dbapi import _Field, _Record

_rec_cache = {}
_no_such_field = object()

__all__ = [
    'Record',
    'Field',
]

def getRecord(name):
    try:
        return _rec_cache[name]
    except KeyError:
        rec = Record(name)
        _rec_cache[name] = rec
        return rec

class Record(_Record):
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

class Field(_Field):
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
