
from __future__ import print_function
import threading, traceback
try:
    import Queue as queue
except ImportError:
    import queue

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
    >>> T.E.wait(1.0)
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
    >>> w.start()
    >>> w.join()
    >>> import threading
    >>> E = threading.Event()
    >>> w = Worker()
    >>> w.start()
    >>> w.add(E.set)
    True
    >>> E.wait(1.0)
    True
    >>> w.join()
    """
    StopWorker = object()

    def __init__(self, max=0):
        super(Worker, self).__init__()
        self._Q = queue.Queue(maxsize=max)
        self.__stop = False
        self.__lock = threading.Lock()


    def join(self, flush=True):
        """Stop accepting new jobs and join the worker thread
        
        Blocks until currently queued work is complete.
        """
        with self.__lock:
            self.__stop = True
        self._Q.put((self.StopWorker,None,None))
        super(Worker, self).join()

    def __len__(self):
        return self._Q.qsize()

    def add(self, func, args=(), kws={}, block=True):
        """Attempt to send a job to the worker.
        
        :returns: True if the job was queued. False if the queue is full,
          or has been joined.
        
        When ``block=True`` then this method will only return
        False if the Worker has been joined.
        """
        with self.__lock:
            if self.__stop is True:
                return False
        try:
            self._Q.put((func,args,kws), block)
            return True
        except queue.Full:
            return False

    def run(self):
        Q = self._Q
        block = True
        try:
            while True:
                F, A, K = Q.get(block)

                if F is self.StopWorker:
                    block = False
                    Q.task_done()
                    continue

                try:
                    F(*A, **K)
                except:
                    print('Error running',F,A,K)
                    traceback.print_exc()
                finally:
                    Q.task_done()
        except queue.Empty:
            pass # We are done now

if __name__=='__main__':
    import doctest
    doctest.testmod()
