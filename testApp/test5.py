
import threading
from devsup.hooks import addHook
from devsup.util import Worker
from devsup.db import IOScanListThread

#(forwardfn, reversefn)
_fns = {
    'none':(lambda x:x, lambda x:x),
    'half':(lambda x:x*2.0, lambda x:x/2.0),
}

ReSync = object()

instances = {}

class UnitWorker(object):
    def __init__(self, name):
        super(UnitWorker,self).__init__()
        self.name = name
        self.scan = IOScanListThread()
        self.scan.force = 0

        self.worker = Worker()

        addHook('AfterIocRunning', self.worker.start)
        addHook('AtIocExit', self.worker.join)

    def add(self, rec, unit, val):
        self.worker.add(self.update, (rec, unit, val))

    def update(self, rec, unit, val):
        F, _ = _fns[unit]
        V = F(val)

        values = {}

        for U,(F,R) in _fns.iteritems():
            values[U] = R(V)

        self.scan.interrupt(reason=values)

class UnitSupport(object):

    raw = True

    def __init__(self, rec, args):
        worker, self.unit = args.split(None, 1)
        try:
            W = instances[worker]
        except KeyError:
            W = UnitWorker(worker)
            instances[worker] = W

        self.worker = W
        W.scan.add(rec)

    def detach(self, rec):
        self.worker.scan.remove(rec)

    def process(self, rec, reason):
        if reason is None:
            self.worker.update(rec, self.unit, rec.VAL)

        else:
            try:
                rec.VAL = reason[self.unit]
                rec.UDF = 0
            except:
                rec.UDF = 1
                raise

build = UnitSupport
