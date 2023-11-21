
import os
import unittest
import tempfile
import time

import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..db import getRecord
from .. import _dbapi
from .. import _init

from .util import IOCHelper

# short-circuit warning from base_registerRecordDeviceDriver()
os.environ['TOP'] = _dbapi.XPYDEV_BASE # external code use devsup.XPYDEV_BASE

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

        with src:
            src.VAL = 42
            self.assertEqual(src.VAL, 42)

        with tgt:
            self.assertEqual(tgt.VAL, 0)

        src.scan(sync=True) # lock and dbProcess() on this thread

        with tgt:
            self.assertEqual(tgt.VAL, 42)

class TestField(IOCHelper):
    db = """
        record(ai, "rec:ai") {
            field(VAL , "4.2")
            field(RVAL, "42")
        }
        record(stringin, "rec:si") {
            field(VAL, "")
        }
        record(waveform, "rec:wf:a") {
            field(FTVL, "DOUBLE")
            field(NELM, "10")
        }
        record(waveform, "rec:wf:s") {
            field(FTVL, "STRING")
            field(NELM, "10")
        }
    """

    def test_ai(self):
        rec = getRecord("rec:ai")

        with rec:
            self.assertEqual(rec.VAL, 4.2)
            self.assertEqual(rec.RVAL, 42)
            rec.VAL = 5.2
            rec.RVAL = 52
            self.assertEqual(rec.VAL, 5.2)
            self.assertEqual(rec.RVAL, 52)

            rec.VAL += 1.0
            self.assertEqual(rec.VAL, 6.2)

    def test_si(self):
        rec = getRecord("rec:si")

        with rec:
            self.assertEqual(rec.VAL, "")

            rec.VAL = "test"
            self.assertEqual(rec.VAL, "test")

            rec.VAL = ""
            self.assertEqual(rec.VAL, "")

            # implicitly truncates
            rec.VAL = "This is a really long string which should be truncated"
            self.assertEqual(rec.VAL, "This is a really long string which shou")

            # TODO: test unicode

    def test_wf_float(self):
        rec = getRecord("rec:wf:a")

        with rec:
            assert_array_almost_equal(rec.VAL, [])

            rec.VAL = numpy.arange(5)
            assert_array_almost_equal(rec.VAL, numpy.arange(5))

            rec.VAL = numpy.arange(10)
            assert_array_almost_equal(rec.VAL, numpy.arange(10))

            with self.assertRaises(ValueError):
                rec.VAL = numpy.arange(15)

            rec.VAL = []
            assert_array_almost_equal(rec.VAL, [])

            # in-place modification
            fld = rec.field('VAL')
            fld.putarraylen(5)
            arr = fld.getarray()
            self.assertEqual(arr.shape, (10,)) # size of NELM
            arr[:5] = numpy.arange(5) # we only fill in the part in use
            arr[2] = 42

            assert_array_almost_equal(rec.VAL, [0, 1, 42, 3, 4])

    def test_wf_string(self):
        rec = getRecord("rec:wf:s")

        with rec:
            assert_array_equal(rec.VAL, numpy.asarray([], dtype='S40'))

            rec.VAL = ["zero", "", "one", "This is a really long string which should be truncated", "", "last"]

            assert_array_equal(rec.VAL,
                                numpy.asarray(["zero", "", "one", "This is a really long string which shoul", "", "last"], dtype='S40'))

class TestLongStringField(IOCHelper):
    db = """
        record(lsi, "rec:lsi") {
            field(SIZV,  128)
            field(SCAN,  "I/O Intr")
        }
        record(lso, "rec:lso") {
            field(SIZV,  128)
            field(DOL,   "rec:lsi.VAL$")
            field(OMSL,  "closed_loop")
        }
    """

    def test_lsilso(self):
        lsi = getRecord("rec:lsi")
        lso = getRecord("rec:lso")

        with lsi:
            self.assertEqual(lsi.VAL, "")

            lsi.VAL = "test"
            self.assertEqual(lsi.VAL, "test")

            lsi.VAL = ""
            self.assertEqual(lsi.VAL, "")

            # does not truncate
            lsi.VAL = "This is a really long string which should NOT be truncated"
            self.assertEqual(lsi.VAL, "This is a really long string which should NOT be truncated")
        
        lso.scan()
        time.sleep(0.01) # The linked value needs a small amount of time to update.

        with lso:
            self.assertEqual(lso.VAL, lsi.VAL)
            
class TestInt64Field(IOCHelper):
    db = """
        record(int64in, "rec:in64") {
            field(SCAN,  "I/O Intr")
        }
        record(int64out, "rec:out64") {
            field(DOL,   "rec:in64.VAL")
            field(OMSL,  "closed_loop")
        }
    """

    def testint64(self):
        in64 = getRecord("rec:in64")
        out64 = getRecord("rec:out64")

        with in64:
            self.assertEqual(in64.VAL, 0)

            in64.VAL = 42
            self.assertEqual(in64.VAL, 42)

            in64.VAL = 0x7FFFFFFFFFFFFFFE
            self.assertEqual(in64.VAL, 0x7FFFFFFFFFFFFFFE)

            in64.VAL = 0x7FFFFFFFFFFFFFFF
            self.assertEqual(in64.VAL, 0x7FFFFFFFFFFFFFFF)
        
        out64.scan()
        time.sleep(0.01) # The linked value needs a small amount of time to update.

        with out64:
            self.assertEqual(out64.VAL, in64.VAL)

class TestDset(IOCHelper):
    db = """
        record(longin, "rec:li") {
            field(DTYP, "Python Device")
            field(INP , "@devsup.test.test_db|TestDset foo bar")
        }
    """

    class Increment(object):
        def process(self, rec, reason):
            rec.VAL += 1
        def detach(self, rec):
            pass

    @classmethod
    def build(klass, rec, args):
        if rec.name()=='rec:li':
            return klass.Increment()
        else:
            raise RuntimeError("Unsupported")

    def test_increment(self):
        rec = getRecord('rec:li')

        with rec:
            self.assertEqual(rec.VAL, 0)
            self.assertEqual(rec.UDF, 1)

        rec.scan(sync=True)

        with rec:
            self.assertEqual(rec.VAL, 1)
            self.assertEqual(rec.UDF, 0)
