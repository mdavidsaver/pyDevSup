
import os
import unittest
import tempfile

import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..db import getRecord
from .. import _dbapi
from .. import _init

__all__ = (
    'IOCHelper',
)

class IOCHelper(unittest.TestCase):
    """Test case run in an IOC. ::

        from devsup.db import getRecord
        from devsup.test.util impmort IOCHelper
        class TestScan(IOCHelper): # sub-class of unittest.TestCase
            db = \"\"\"
                record(longout, foo) {}
            \"\"\"
            autostart = True

            def test_link(self):
                rec = getRecord('foo')
                with rec: # dbScanLock()
                    self.assertEqual(rec.VAL, 0)
    """
    # DB definition to be used.  May include eg. 'record(ai, "blah") {}'
    db = None
    # Whether to automatically run iocInit() before test methods
    # whether iocInit() has been called
    autostart = True
    running = False

    def setUp(self):
        print("testdbPrepare()")
        _dbapi._UTest.testdbPrepare()
        _init(iocMain=False) # load base.dbd

        if self.db is not None:
            with tempfile.NamedTemporaryFile() as F:
                F.write(self.db.encode('ascii'))
                F.flush()
                _dbapi.dbReadDatabase(F.name)

        if self.autostart:
            self.iocInit()

    def tearDown(self):
        self.iocShutdown();
        print("testdbCleanup()")
        _dbapi.initHookAnnounce(9999) # our magic/fake AtExit hook
        _dbapi._UTest.testdbCleanup()

    def iocInit(self):
        """If not autostart, then this must be called before runtime database access is possible
        """
        if not self.running:
            print("testIocInitOk")
            _dbapi._UTest.testIocInitOk()
            self.running = True

    def iocShutdown(self):
        """Call to stop IOC scanning processes.  Happens automatically during test tearDown
        """
        if self.running:
            print("testIocShutdownOk")
            _dbapi._UTest.testIocShutdownOk()
            self.running = False
