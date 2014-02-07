# -*- coding: utf-8 -*-

import logging
LOG = logging.getLogger(__name__)

import devsup.ptable as PT

class SumTable(PT.TableBase):
    A = PT.Parameter()
    B = PT.Parameter()
    C = PT.Parameter()
    S = PT.Parameter(iointr=True)
    
    inputs = PT.ParameterGroup([A,B])

    @C.onchange
    def newC(self):
        LOG.debug("C is %s", self.C.value)

    @inputs.anynotvalid
    def inval(self):
        print self.A.isvalid, self.B.isvalid
        LOG.debug("%s.update inputs not valid", self.name)
        self.S.value = None
        self.S.notify()

    @inputs.allvalid
    def update(self):
        if not all(map(lambda P:P.isvalid, [self.A, self.B])):
            self.inval()
            return
        self.S.value = self.A.value + self.B.value
        LOG.debug("%s.S = %s", self.name, self.S.value)
        self.S.notify()
