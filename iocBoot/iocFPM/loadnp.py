# -*- coding: utf-8 -*-
"""
Waveform device support which reads
from a .npy file.

Intended to load test input data
"""

import numpy

class Device(object):
    def __init__(self, rec, fname):
        if not fname:
            return
        print 'Load', fname,'into',rec.NAME
        data = numpy.load(fname)
        assert len(data.shape)==1, 'only 1D supported'
        val = rec.field('VAL')
        vlen = min(len(val), len(data))
        val.getarray()[:vlen] = data[:vlen]
        val.putarraylen(vlen)

    def process(self, rec, reason=None):
        pass

build = Device
