try:
    import _dbapi
except ImportError:
    import _nullapi as _dbapi
from _dbapi import _Field, _Record

_rec_cache = {}

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
        self._fld_cache = {}
    def field(self, name):
        """Lookup field in this record
        
        fld = rec.field('HOPR')
        """
        try:
            return self._fld_cache[name]
        except KeyError:
            fld = Field("%s.%s"%(self.name(), name))
            self._fld_cache[name] = fld
            return fld

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

    def __repr__(self):
        return 'Field("%s.%s")'%self.name()
