from __future__ import print_function

import datetime

class MySup(object):
    def __init__(self, rec, args):
        pass
    def process(self, rec, reason):
        rec.VAL = 1+rec.VAL
        # could also pass in float or tuple (sec, nanosec)
        rec.setTime(datetime.datetime.now())
    def detach(self, rec):
        pass

build = MySup
