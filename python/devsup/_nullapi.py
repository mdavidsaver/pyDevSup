
def verinfo():
    """(VER, REV, MOD, PATH, "site") = verinfo()
    
    Query EPICS version information
    """
    return (0,0,0,0.'invalid')

class Record(object):
    """Handle for record operations
    
    r = Record("rec:name")
    """

    def __init__(self, rec):
        pass
    def name(self):
        """Record name
        """
    def info(self, key):
        """info(key)
        info(key, default)
        
        Lookup record info tag.  If no default
        is provided then an exception is raised
        if the info key does not exist.
        """
    def infos(self):
        """Return a dictionary of all info tags
        for this record
        """

    def scan(self, sync=False):
        """scan(sync=False)
        
        Scan this record.  If sync is False then a
        scan request is queued.  If sync is True then the record
        is scannined immidately on the current thread.
        """

def Field(object):
    """Handle for field operations
    
    f = Field("rec:name.HOPR")
    """
    def __init__(self, fld):
        pass
    def name(self):
        """("rec", "FLD") = name()
        """
    def fieldinfo(self):
        """(type, size, #elements) = fieldinfo()
        
        Type is DBF type code
        size is number of bytes to start a single element
        #elements is the maximum number of elements the field can hold
        """

    def getval(self):
        """Fetch the current field value as a scalar.
        
        Returns Int, Float, or str
        """

    def putval(self, val):
        """Update the field value
        
        Must be an Int, Float or str
        """
