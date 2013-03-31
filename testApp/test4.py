
import threading

AsyncComplete = object()

class Counter(object):
    def __init__(self, rec, args):
        self.val = rec.field('VAL')
        self.nextval = None
        self.timer = None
    def detach(self, rec):
        if self.timer:
            self.timer.cancel()

    def process(self, rec, reason):
        if reason is AsyncComplete:
            self.val.putval(self.nextval)
            
        else:
            self.nextval = self.val.getval()+1
            self.timer = threading.Timer(0.2, rec.asyncFinish, kwargs={'reason':AsyncComplete})
            rec.asyncStart()
            self.timer.start()


build = Counter
