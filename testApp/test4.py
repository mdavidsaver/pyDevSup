
import threading

AsyncComplete = object()

class Counter(object):
    def __init__(self, rec, args):
        self.nextval = None
        self.timer = None
    def detach(self, rec):
        if self.timer:
            self.timer.cancel()

    def process(self, rec, reason):
        if reason is AsyncComplete:
            rec.VAL = self.nextval

        else:
            self.nextval = rec.VAL+1
            self.timer = threading.Timer(0.2, rec.asyncFinish, kwargs={'reason':AsyncComplete})
            rec.asyncStart()
            self.timer.start()


build = Counter
