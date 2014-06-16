
class _Record(object):
    """Handle for record operations
    
    r = _Record("rec:name")
    """

    def __init__(self, rec):
        pass
    def name(self):
        """Record name string.
        
        >>> R = getRecord("my:record:name")
        >>> R.name()
        "my:record:name"
        """
    def rtype(self):
        """Record type name string.
        
        >>> R = getRecord("my:record:name")
        >>> R.type()
        "longin"
        """
    def isPyRecord(self):
        """Is this record using Python device support.
        
        :rtype: bool
        """
    def info(self, key):
        """info(key [,default])
        
        :rtype: str
        :throws: KeyError
        
        Lookup record info tag.  If no default
        is provided then an exception is raised
        if the info key does not exist.
        """
    def infos(self):
        """Return a dictionary of all info tags
        for this record
        """

    def setSevr(self, sevr=3, stat=15):
        """setSevr(sevr=INVALID_ALARM, stat=COMM_ALARM)
        
        Signal a new alarm condition.  The effect of this
        call depends on the current alarm condition.
        
        See :c:func:`recGblSetSevr` in EPICS Base.
        """

    def scan(self, sync=False, reason=None, force=0):
        """Scan this record.
        
        :param sync: scan in current thread (``True``), or queue to a worker (``False``).
        :param reason: Reason object passed to :meth:`process <DeviceSupport.process>` (sync=True only)
        :param force: Record processing condtion (0=Passive, 1=Force, 2=I/O Intr)
        :throws: ``RuntimeError`` when ``sync=True``, but ``force`` prevents scanning.
        
        If ``sync`` is False then a scan request is queued to run in another thread..
        If ``sync`` is True then the record is scanned immediately on the current thread.
        
        For ``reason`` argument must be used in conjunction with ``sync=True``
        on records with Python device support.  This provides a means
        of providing extra contextual information to the record's
        :meth:`process <DeviceSupport.process>` method.
        
        ``force`` is used to decide if the record will actually be processed,
        ``force=0`` will only process records with SCAN=Passive.
        ``force=1`` will process any record if at all possible.
        ``force=2`` will only process records with Python device support and
        SCAN=I/O Intr.
        
        ..  important::
          It is **never** safe to use ``sync=True`` while holding record locks,
          including from within a *process* method.
        """

    def asyncStart(self):
        """Start asynchronous processing
        
        This method may be called from a device support
        :meth:`process <DeviceSupport.process>` method
        to indicate that processing will continue
        later.
        
        ..  important::
          This method is **only** safe to call within a *process* method.
        """
    def asyncFinish(self, reason=None):
        """Indicate that asynchronous processing can complete
        
        Similar to :meth:`scan`.  Used to conclude asynchronous
        process started with :meth:`asyncStart`.
        
        Processing is completed on the current thread.
        
        ..  important::
          This method should **never** be called within
          a :meth:`process <DeviceSupport.process>` method,
          or any other context where a Record lock is held.
          Doing so will result in a deadlock.

        Typically a *reason* will be passed to *process* as a way
        of indicating that this is the completion of an async action. ::
        
          AsyncDone = object()
          class MySup(object):
            def process(record, reason):
              if reason is AsyncDone:
                record.VAL = ... # store result
              else:
                threading.Timer(1.0, record.asyncFinish, kwargs={'reason':AsyncDone})
                record.asyncStart()
        """

class _Field(object):
    """Handle for field operations
    
    f = Field("rec:name.HOPR")
    
    Field objects implement the buffer protocol.
    """
    def __init__(self, fld):
        pass
    def name(self):
        """Fetch the record and field names.
        
        >>> FLD = getRecord("rec").field("FLD")
        >>> FLD.name()
        ("rec", "FLD")
        """
    def fieldinfo(self):
        """(type, size, #elements) = fieldinfo()
        
        Type is DBF type code
        size is number of bytes to start a single element
        #elements is the maximum number of elements the field can hold
        """

    def getval(self):
        """Fetch the current field value as a scalar.
        
        :rtype: int, float, or str
        
        Returned type depends of field DBF type.
        An ``int`` is returned for CHAR, SHORT, LONG, and ENUM.
        A ``float`` is returned for FLOAT and DOUBLE.
        A ``str`` is returned for STRING.
        """

    def putval(self, val):
        """Update the field value
        
        Must be an Int, Float or str.  Strings will be truncated to 39 characters.
        """

    def getarray(self):
        """Return a numpy ndarray refering to this field for in-place operations.
        
        The dtype of the ndarray will correspond to the field's DBF type.
        Its size will be the **maximum** number of elements.
        
        .. important::
          It is only safe to read or write to this ndarray while the record
          lock is held (ie withing :meth:`process <DeviceSupport.process>`).
        """

    def getarraylen(self):
        """Return the number of active elements for the field.
        
        >>> F = Field(...)
        >>> assert len(F)>=F.getarraylen()
        """

    def putarraylen(self, len):
        """Set the number of active elements in field's array.

        Requires that the underlying field be an array.
        Must be greater than one and less than or equal to the maximum length of the field.
        """

    def getAlarm(self):
        """Returns a tuple (severity, status) with the condition of the linked field.
        
        Only works for fields of type DBF_INLINK.
        """

    def __len__(self):
        """Returns the maximum number of elements which may be stored in the field.

        This is always 1 for scalar fields.
        """

_hooks = {}
