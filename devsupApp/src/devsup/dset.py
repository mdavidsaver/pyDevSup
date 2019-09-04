# -*- coding: utf-8 -*-
"""
Some general purpose device supports
"""

from __future__ import print_function

import logging
_log = logging.getLogger(__name__)

import time
from .hooks import initHook

class AsyncOffload(object):
    """A device support which off-loads processing to a worker thread.

    >>> class ArrSum(AsyncOffload):
        inputs = {'A':'one', 'B':'two'}
        outputs= {'VALA':'result'}

        def inThread(one=0, two=0):
            return {'result':one+two}


    The attribute "worker" may be set to an instance of Worker, either
    statically, or in the derived class constructor (before calling the Base
    class constructor).
    If this attribute remains None, then a new Worker thread is created for
    each record instance.

    The attribute "scan" may similarly be set to an instance of IOScanListBlock.
    
    The "inputs" and "outputs" dictionary exist to map record Field names
    to internal names.  This also serves to indicate which fields the inThread
    methods may use.  This avoid processing of unnecesary (array) fields.
    """

    worker = None
    scan = None

    inputs = {}
    outputs = {}
    timefld = None

    def __init__(self, rec, link):
        self.rec, self.link = rec, link

        if self.worker is None:
            from devsup.util import Worker
            self.worker = Worker(max=1)

            @initHook('AtIocExit')
            def _exit():
                print('stop worker for', rec.NAME)
                self.worker.join()

            self.worker.start()

        assert self.worker is not None, "Offload requires a worker thread"

        I = []
        for fld,name in self.inputs.items():
            F = rec.field(fld)
            I.append((fld, name, F.fieldinfo()[2]!=1))

        O = list(self.outputs.items())

        self._inputs, self._outputs = I, O

    def allowScan(self, rec):
        if self.scan:
            return self.scan.add(rec)

    def detach(self, rec):
        pass

    def process(self, rec, reason=None):
        if reason is None:
            self._tstart = time.time()
            
            V = {}
            for fld,name,arr in self._inputs:
                val = getattr(rec, fld)
                if arr:
                    val = val.copy()
                V[name] = val

            self.worker.add(self._wrap, args=(rec,), kws=V)

            rec.asyncStart()

        else:
            result = reason
            if result['ok']:
                for fld,name in self._outputs:
                    setattr(rec, fld, result.get(name, 0))

                sevr = result.get('severity', 0)
                if sevr:
                    rec.setSevr(sevr)

            else:
                rec.setSevr()

            self._tend = time.time()
            if self.timefld:
                setattr(rec, self.timefld, self._tend-self._tstart)

    def _wrap(self, rec, **kws):
        try:
            result = self.inThread(**kws)
            result.setdefault('ok', True)
            rec.asyncFinish(reason=result)
        except:
            _log.exception('Error while processing %s', rec.NAME)
            rec.asyncFinish(reason={'ok':False})

    def inThread(self, **kws):
        return {'ok':False}
