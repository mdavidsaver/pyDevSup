
import weakref
import threading, time
from devsup.hooks import addHook

insts = {}

def done(obj):
    print obj,'Expires'

_tracking = {}
def track(obj):
    W = weakref.ref(obj, done)
    print 'track',obj,'with',W
    _tracking[id(obj)] = W

class Driver(threading.Thread):
    def __init__(self, name):
        super(Driver,self).__init__()
        track(self)
        self.name = name
        self._lock = threading.Lock()
        self._recs = set()
        self._run = True
        self._stop = threading.Event()
        self.value = 0
        addHook('AfterIocRunning', self.start)
        addHook('AtIocExit', self.stop)

    def stop(self):
        print 'Stopping driver',self.name
        with self._lock:
            self._run = False
        self._stop.wait()
        print 'Finished with',self.value

    def addrec(self, rec):
        with self._lock:
            self._recs.add(rec)
    def delrec(self, rec):
        with self._lock:
            self._recs.remove(rec)

    def run(self):
        try:
            while self._run:
                time.sleep(1.0)
                with self._lock:
                    val = self.value
                    self.value += 1
                    for R in self._recs:
                        R.record.scan(sync=True, reason=val)
        finally:
            self._stop.set()

def addDrv(name):
    print 'Create driver',name
    insts[name] = Driver(name)

class Device(object):
    def __init__(self, rec, drv):
        track(self)
        self.driver, self.record = drv, rec
        self.driver.addrec(self)
        self.val = rec.field('VAL')
    def detach(self, rec):
        self.driver.delrec(self)
    def process(self, rec, data):
        if data is None:
            print rec,'Someone processed me?'
        else:
            print rec,'update to',data
            self.val.putval(data)

def build(rec, args):
    drv = insts[args]
    return Device(rec, drv)
