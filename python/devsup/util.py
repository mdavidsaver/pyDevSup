
import threading, traceback

class Worker(threading.Thread):
    """A threaded work queue.

    >>> w = Worker()
    >>> w = Worker()
    >>> w.start()
    >>> w.stop()
    >>> import threading
    >>> E = threading.Event()
    >>> w = Worker()
    >>> w.start()
    >>> w.add(E.set)
    >>> E.wait(1.0)
    True
    >>> w.stop()
    """
    def __init__(self, max=0):
        super(Worker, self).__init__()
        self._run, self._stop = False, None
        self._lock = threading.Lock()
        self._update = threading.Condition(self._lock)
        self.maxQ, self._Q = max, []

    def running(self):
        with self._lock:
            return self._run and not self._stop

    def start(self):
        with self._lock:
            if self._run or self._stop:
                return
            super(Worker, self).start()
            self._run = True

    def stop(self, flush=True):
        self._update.acquire()
        try:
            if self._stop:
                raise RuntimeError("Someone else is already trying to stop me")

            self._stop = threading.Event()
            self._update.notify()

            self._update.release()
            try:
                self._stop.wait()
            finally:
                self._update.acquire()

            self._stop = None
            assert not self._run
            
            if flush:
                self._Q = []

        finally:
            self._update.release()

    def __len__(self):
        with self._lock:
            return len(self._Q)

    def add(self, func, args=(), kws={}):
        with self._lock:
            if not self._run or self._stop:
                return
            elif self.maxQ>0 and len(self._Q)>=self.maxQ:
                raise RuntimeError('Worker queue full')

            self._Q.append((func,args,kws))
            self._update.notify()

    def run(self):
        self._update.acquire()
        try:
            assert self._run

            while True:

                while self._stop is None and len(self._Q)==0:
                    self._update.wait()

                if self._stop is not None:
                    break

                F, A, K = self._Q.pop(0)

                self._update.release()
                try:
                    F(*A,**K)
                except:
                    print 'Error running',F,A,K
                    traceback.print_exc()
                finally:
                    self._update.acquire()

            self._run = False
            self._stop.set()

        finally:
            self._update.release()

if __name__=='__main__':
    import doctest
    doctest.testmod()
