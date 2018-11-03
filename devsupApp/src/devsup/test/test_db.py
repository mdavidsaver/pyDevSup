
import os
import unittest
import tempfile

from ..db import getRecord
from .. import _dbapi
from .. import _init

# short-circuit warning from _dbapi._init()
os.environ['TOP'] = _dbapi.XPYDEV_BASE

class IOCHelper(unittest.TestCase):
    db = None
    autostart = running = False
    def setUp(self):
        print("testdbPrepare()")
        _dbapi._UTest.testdbPrepare()
        _init(iocMain=False) # load base.dbd

        if self.db is not None:
            with tempfile.NamedTemporaryFile() as F:
                F.write(self.db)
                F.flush()
                _dbapi.dbReadDatabase(F.name)

        if self.autostart:
            self.iocInit()

    def tearDown(self):
        self.iocShutdown();
        print("testdbCleanup()")
        _dbapi._UTest.testdbCleanup()

    def iocInit(self):
        if not self.running:
            print("testIocInitOk")
            _dbapi._UTest.testIocInitOk()
            self.running = True

    def iocShutdown(self):
        if self.running:
            print("testIocShutdownOk")
            _dbapi._UTest.testIocShutdownOk()
            self.running = False

class TestIOC(IOCHelper):
    def test_base(self):
        pass

    def test_start(self):
        self.iocInit()
        self.iocShutdown()

    def test_db(self):
        with tempfile.NamedTemporaryFile() as F:
            F.write('record(longin, "test") {}\n')
            F.flush()
            _dbapi.dbReadDatabase(F.name)

        rec = getRecord("test")
        self.assertEqual(rec.VAL, 0)
        rec.VAL = 5
        self.assertEqual(rec.VAL, 5)

class TestScan(IOCHelper):
    db = """
        record(longout, src) {
            field(OUT, "tgt PP")
        }
        record(longin, "tgt") {}
    """
    autostart = True

    def test_link(self):
        src, tgt = getRecord('src'), getRecord('tgt')

        src.VAL = 42
        self.assertEqual(src.VAL, 42)
        self.assertEqual(tgt.VAL, 0)
        src.scan(sync=True)
        self.assertEqual(tgt.VAL, 42)
