
import numpy as np
import scipy as sp
from numpy.random import randint, uniform

class WfSup(object):
    def __init__(self, rec, args):
        self.arr = rec.field('VAL').getarray()
        self.x = np.arange(rec.NELM)

        self.phase = 0.0

    def detach(self, rec):
        pass

    def process(self, rec, reason):
        pha = self.phase*np.pi/180.0
        self.phase = np.fmod(self.phase+10.0, 360.0)

        N=randint(1, rec.NELM)
        
        val=self.arr[:N]
        x=self.x[:N]

        # calculate inplace: uniform(0.5,2.0)*sin(pha*x)+2
        val[:] = x
        val[:] *= pha
        np.sin(val, out=val)
        val[:]*=uniform(0.5,2.0)
        val[:]+=2
        
        self.NORD = N

def build(rec, args):
    return WfSup(rec, args)
