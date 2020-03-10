Support Modules
===============

An EPICS Record definition for most record types will
include setting the DTYP and INP or OUT fields.
An example with the longin would be: ::

  record(longin, "instance:name") {
    field(DTYP, "Python Device")
    field(INP , "@pymodule some other string")
  }

Or equivalently: ::

  record(longin, "instance:name") {
    field(DTYP, "Python Device")
    field(INP , "@some other string")
    info("pySupportMod", "pymodule")
  }

This minimal example will attempt to import a Python
module named 'pymodule'.  This module is expected
to provide a :func:`build` function.
Which will be called with a :class:`Record <devsup.db.Record>` instance
and the string "some other string".

:func:`build` Function
----------------------

.. function:: build(record, args)

  Called when the IOC requests a device support instance
  for the given record.  This function should return
  an object providing the methods of a
  :class:`DeviceSupport`.

  :param record: The :class:`Record <devsup.db.Record>` instance to which this support
    will be attached.  May be used to introspect and
    query initial field values.
  :param args: The remainder of the INP or OUT field string.
  
  ::
  
    def build(record, args):
      print 'Need device support for', record.name()
      print 'Provided:', args
      raise RuntimeError("Not support found!")

:class:`DeviceSupport` Interface
--------------------------------

.. class:: DeviceSupport

  ``DeviceSupport`` is not a class.  Rather it is a description
  of the methods which all Python device support instances must provide.
  These methods will be called during the course of IOC processing.

  Exceptions raised by these methods are printed to the IOC console,
  but will otherwise be ignored.

  The module :mod:`devsup.interfaces` provides a Zope Interface
  definition by this name which may be referenced.

  .. attribute:: raw

    A boolean value indicating whether this device support
    uses "raw" access.  A Raw support module will update
    the VAL field even if the recordtype has an RVAL field
    (eg. ai/ao, mbbi/mbbo).

    Omitting this attribute is the same as False.

  .. method:: process(record, reason)

    :param record: :class:`Record <devsup.db.Record>` from which the request originated.
    :param reason: ``None`` or an object provided when processing was requested.

    This method is called whenever the associated record needs to be updated
    in response to a request.  The source of this request is typically determined
    by the record's SCAN field.

    The actions taken by this method will depend heavily on the application.
    Typically this will include reading or writing values from fields.
    Record fields can be access through the provided ``Record`` instance.

  .. method:: detach(record)

    :param record: :class:`Record <devsup.db.Record>` from which the request originated.

    Called when a device support instance is being dis-associated
    from its Record.  This will occur when the IOC is shutdown
    or the INP or OUT field of a record is modified.
    
    No further calls to this object will be made in relation
    to *record*.

  .. method:: allowScan(record)

    :param record: :class:`Record <devsup.db.Record>` from which the request originated.
    :rtype: bool or Callable
  
    Called when an attempt is made to set the record's SCAN field
    to "I/O Intr" either at startup, or during runtime.
    To permit this the method must return an object which evaluates to *True*.
    If not then the attempt will fail and SCAN will revert to
    "Passive".

    If a callable object is returned, then it will be invoked
    when SCAN is changed again, or just before :meth:`detach`
    is called.

    This method will typically be implemented using the
    ``add`` method of an I/O scan list object.
    (:meth:`IOScanListBlock <devsup.db.IOScanListBlock.add>`
    or :meth:`IOScanListThread <devsup.db.IOScanListThread.add>`) ::
    
      class MySup(object):
        def __init__(self):
          self.a_scan = devsup.db.IOScanListThread()
        def allowScan(self, record):
          return self.a_scan.add(record)

    Which in most cases can be abbriviated to ::

      class MySup(object):
        def __init__(self):
          self.a_scan = devsup.db.IOScanListThread()
          self.allowScan = self.a_scan.add

Example
-------

A simple counter.  The processing action is to increment the value
of the VAL field.
The following code should be placed in a file named *counter.py*
which should be placed in the Python module import path. ::

  class MySupport(object):
    def detach(self, record):
      pass # no cleanup needed

    def allowScan(self, record):
      return False # I/O Intr not supported

    def process(self, record, reason):
      record.VAL = record.VAL + 1
      try:
        record.UDF = 0
      except AttributeError:
        pass # not all record types implement this

  def build(record, args):
    if not args.startswith('hello'):
      raise RuntimeError('%s is not friendly.'%record)
    return MySupport()

This support code can then be referenced from records. ::

  record(longin, "my:int:counter") {
    field(DTYP, "Python Device")
    field(INP , "@counter hello world")
  }
  
The following will fail to associate. ::

  record(longin, "my:int:counter") {
    field(DTYP, "Python Device")
    field(INP , "@counter do what I say")
  }

If a shorter INP link string is necessary, or to prevent
runtime switching of modules, the module name may also
be given in the *pySupportMod* info() tag. ::

  record(longin, "my:int:counter") {
    field(DTYP, "Python Device")
    field(INP , "@hello world")
    info("pySupportMod", "counter")
  }
