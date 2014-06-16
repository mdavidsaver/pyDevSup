Getting Started
===============

Counter
-------

Consider a simple EPICS database with one record.  Call it :download:`cntrec.db <../testApp/cntrec.db>`

.. literalinclude:: ../testApp/cntrec.db
  
This is creating a single record which will use the "Python Device" support code (aka this package).
It will attempt to scan (call the process method) one a second.
The *INP* field is parsed and the first work identifies the Python module which will provide
the logic behind this record (everything after the first word is passed to the module :py:func:`build` function.

Now create :download:`cntrec.db <../testApp/cntmod.py>` with the following.

.. literalinclude:: ../testApp/cntmod.py

This module is expected to provide a special callable :py:func:`build`.
We also provide a constructor and method :py:meth:`detach <DeviceSupport.detach>`
which don't do anything.
The :py:meth:`process <DeviceSupport.process>` method increments the *VAL* field of the attached :py:class:`Record <devsup.db.Record>`.

Start this IOC with. ::

  $ ./bin/linux-x86_64/softIocPy2.7 -d cntrec.db
  Starting iocInit
  ...
  iocRun: All initialization complete
  epics>dbl
  test:count
  epics>

Now in another terminal run.::

  $ camonitor test:count
  ...
  test:count                     2014-06-16 16:48:22.891825 9  
  test:count                     2014-06-16 16:48:23.891967 10  
  test:count                     2014-06-16 16:48:24.892137 11  
  test:count                     2014-06-16 16:48:25.892286 12
  
It may be necessary to run *export EPICS_CA_ADDR_LIST=localhost* first.

Additional examples and applications
------------------------------------

This module comes with several examples in *testApp* as well as three complete applications.

logApp
  Observes a line based text file as new lines are appended.
  Writes each line to a charactor array PV.
  Special handling of caPutLog files.

pidMonApp
  Monitors the PID file created by a UNIX daemon.

weatherApp
  Retreives weather reports via the *pymetar* module.
