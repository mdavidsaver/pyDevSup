
from __future__ import print_function
import threading, traceback

class StoppableThread(threading.Thread):
    """A thread which can be requested to stop.
    
    The thread run() method should periodically call the shouldRun()
    method and return if this yields False.
    
    >>> class TestThread(StoppableThread):
    ...     def __init__(self):
    ...         super(TestThread,self).__init__()
    ...         self.E=threading.Event()
    ...     def run(self):
    ...         import time
    ...         self.cur = threading.current_thread()
    ...         self.E.set()
    ...         while self.shouldRun():
    ...             time.sleep(0.01)
    >>> T = TestThread()
    >>> T.start()
    >>> T.E.wait()
    True
    >>> T.cur is T
    True
    >>> T.join()
    >>> T.is_alive()
    False
    """
    def __init__(self, max=0):
        super(StoppableThread, self).__init__()
        self.__stop = True
        self.__lock = threading.Lock()

    def start(self):
        with self.__lock:
            self.__stop = False

        super(StoppableThread, self).start()

    def join(self):
        with self.__lock:
            self.__stop = True
       
        super(StoppableThread, self).join()

    def shouldRun(self):
        with self.__lock:
            return not self.__stop

class Worker(threading.Thread):
    """A threaded work queue.

    >>> w = Worker()
    >>> w = Worker()
    >>> w.start()
    >>> w.join()
    >>> import threading
    >>> E = threading.Event()
    >>> w = Worker()
    >>> w.start()
    >>> w.add(E.set)
    >>> E.wait(1.0)
    True
    >>> w.join()
    """
    def __init__(self, max=0):
        super(Worker, self).__init__()
        self.__stop = None
        self.__lock = threading.Lock()
        self.__update = threading.Condition(self.__lock)
        self.maxQ, self._Q = max, []


    def join(self, flush=True):
        self.__update.acquire()
        try:
            if self.__stop is not None:
                raise RuntimeError("Someone else is already trying to stop me")

            self.__stop = threading.Event()
            self.__update.notify()

            self.__update.release()
            try:
                self.__stop.wait()
            finally:
                self.__update.acquire()

            self.__stop = None
            
            if flush:
                self._Q = []

        finally:
            self.__update.release()

    def __len__(self):
        with self.__lock:
            return len(self._Q)

    def add(self, func, args=(), kws={}):
        with self.__lock:
            if self.__stop is not None:
                return
            elif self.maxQ>0 and len(self._Q)>=self.maxQ:
                raise RuntimeError('Worker queue full')

            self._Q.append((func,args,kws))
            self.__update.notify()

    def run(self):
        self.__update.acquire()
        try:

            while True:

                while self.__stop is None and len(self._Q)==0:
                    self.__update.wait()

                if self.__stop is not None:
                    break

                F, A, K = self._Q.pop(0)

                self.__update.release()
                try:
                    F(*A,**K)
                except:
                    print('Error running',F,A,K)
                    traceback.print_exc()
                finally:
                    self.__update.acquire()

            self.__stop.set()
        except:
            traceback.print_exc()
        finally:
            self.__update.release()

if __name__=='__main__':
    import doctest
    doctest.testmod()
