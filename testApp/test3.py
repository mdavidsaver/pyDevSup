from __future__ import print_function

import numpy as np
import scipy as sp
from numpy.random import randint, uniform

class WfSup(object):
    def __init__(self, rec, args):
        self.fld = rec.field('VAL')
        self.arr = self.fld.getarray()
        assert(len(self.fld)==rec.NELM)
        self.x = np.arange(len(self.fld))

        self.phase = 0.0
        print("start",self)

    def detach(self, rec):
        pass

    def process(self, rec, reason):
        pha = self.phase*np.pi/180.0
        self.phase = np.fmod(self.phase+10.0, 360.0)

        N=randint(1, len(self.fld))
        
        val=self.arr[:N]
        x=self.x[:N]

        # calculate inplace: uniform(0.5,2.0)*sin(pha*x)+2
        val[:] = np.sin(x*pha)*uniform(0.5,2.0) + 2

        self.fld.putarraylen(N)

def build(rec, args):
    return WfSup(rec, args)
