
import numpy as np
import scipy as sp
from numpy.random import randint, uniform

class WfSup(object):
    def __init__(self, rec, args):
        self.VAL = rec.field('VAL')
        self.NORD = rec.field('NORD')
        self.nelm = rec.field('NELM').getval()

        self.arr = self.VAL.getarray()
        self.x = np.arange(self.nelm)

        self.phase = 0.0

    def detach(self, rec):
        pass

    def process(self, rec, reason):
        pha = self.phase*np.pi/180.0
        self.phase += 10.0
        if self.phase>=360.0:
            self.phase == 360.0

        N=randint(1,self.nelm)
        
        val=self.arr[:N]
        x=self.x[:N]

        val[:] = pha*x
        np.sin(val, out=val)
        val[:]*=uniform(0.5,2.0)
        val[:]+=2
        
        self.NORD.putval(N)

def build(rec, args):
    return WfSup(rec, args)
