# -*- coding: utf-8 -*-
"""
Processing of Filling Pattern Monitor digitizer data.
"""

import logging
_log = logging.getLogger(__name__)

import time

import numpy
from scipy import signal

from devsup.util import Worker
from devsup.hooks import initHook
from devsup.dset import AsyncOffload

bucketPturn = 1320 # NSLS2
sampPbucket = 16 # ~= 8GHz/500MHz
sampPturn = bucketPturn*sampPbucket

# Since digitizer sample clock is not sync'd
# to the signal the bunch phase will change
# over the course of the trace.
#
# To compensate for this we should resample w/ interpolation,
# but this is expensive and may distort the signal.
#
# Instead make a coarse correction by periodically dropping
# a sample to simulate a sync'd sample clock
#
# This lets us pretend that there are exactly 16 samples per
# period of the signal

Fsamp = 8e9 # ADC samples frequency
Fsig =499.68e6 # frequency of signal being sampled
mult = Fsamp/Fsig # samples per signal period
nPb = int(mult)
assert nPb==sampPbucket
remd = mult-nPb
assert remd<0.5 and remd>0, "assumption violated" # remd==0 drops no samples, remd>0.5 should add samples
delta = int(numpy.round(nPb/remd))
print 'fpm delta mult', nPb, 'frac', delta

def deleteEveryNth(arr, N):
    '''Optimized version of

    return numpy.delete(arr, numpy.arange(0, len(arr), N))
    '''
    iS = range(0, len(arr), N)
    parts = [None]*len(iS)
    for e,i in enumerate(iS):
        parts[e] = arr[i+1:i+N]
    return numpy.concatenate(parts)

class Dev(AsyncOffload):
    inputs = {
        'A':'raw', # array copy
        'B':'nskip',
        'C':'thres',
        'D':'scale',
        'E':'offset',
    }
    outputs = {
        'VALA':'fill', # array copy
        'VALB':'phase', # array copy
        'VALD':'nwbeam',
        'VALE':'total',
        'VALF':'minv',
        'VALG':'maxv',
        'VALH':'mini',
        'VALI':'maxi',
        'VALJ':'pvar',
        'VALK':'turns',
    }
    timefld = 'VALC' # store execution time


    def inThread(self, raw=None, nskip=0, thres=0.01, scale=1.0, offset=0, **kws):
        raw = signal.detrend(raw[nskip:], type='constant')

        # remove every delta'th sample
        raw = deleteEveryNth(raw, delta)

        # truncate to whole turns
        turns = len(raw)/sampPturn
        raw = raw[:turns*sampPturn]

        # determine buckets w/ beam
        val = raw.reshape(turns, bucketPturn, sampPbucket)

        # Which buckets have beam?
        # average over turns
        # find buckets where peak to peak voltage exceeds threshold
        hasbeam = val.mean(0).ptp(1)>thres
        nwbeam = hasbeam.sum()

        if nwbeam>0:
            # fine tune sync.
            # find peak position for each turn by averaging all buckets w/ beam
            peaksamp = val[:,hasbeam,:].argmax(2).mean(1)
            turnshift = numpy.round(peaksamp).astype(numpy.uint8)

            turnval = raw.reshape(turns, sampPturn)

            # shift turns to align peak with turn zero.
            ## Warning, this mixes signals for buckets 0 and 1319 in adjecent turns
            turn0shift = turnshift[0]

            for i in range(16):
                turnval[turnshift==i,:] = numpy.roll(turnval[turnshift==i,:], turn0shift-i, axis=1)

            val = turnval.reshape(turns, bucketPturn, sampPbucket)

        # sum samples over each RF bucket
        # offset compensates for summing of digitizer noise
        Ival =  numpy.abs(val).sum(2) - offset

        # after subtraction the "current" may be negative, but this
        # will mess up the automatic re-scaling, so force non-negative
        Ival[Ival<0] = 0.0

        fillbyturn = Ival.mean(1) # average buckets

        fill = Ival.mean(0) # average turns

        if scale>0.0:
            S1 = scale/fill.sum()
            fill *= S1
            S2 = scale/fillbyturn[0]
            fillbyturn *= S2
            #print 'factor', S1, S2
        else:
            fill = numpy.zeros(fill.shape, dtype=fill.dtype)
            fillbyturn = numpy.zeros(fillbyturn.shape, dtype=fillbyturn.dtype)

        phase = val.mean(1).mean(0) # sum over turns and buckets to get the approx. bunch signal shape

        if nwbeam>0:
            bfill = fill[hasbeam]
            bindx = numpy.arange(fill.shape[0])[hasbeam]
            pvar = (bfill.max()-bfill.min())/(bfill.max()+bfill.min())*100.0
        else:
            bfill = numpy.asarray([0.0])
            bindx = numpy.asarray([0])
            pvar = 0.0

        #print 'ellapsed 2 %.03f'%(time.time()-TS)
        return {
            'ok':True,
            'fill':fill,
            'turns':fillbyturn,
            'phase':phase,
            'nwbeam':hasbeam.sum(),
            'total':fill.sum(),
            'minv':bfill.min(),
            'maxv':bfill.max(),
            'mini':bindx[bfill.argmin()],
            'maxi':bindx[bfill.argmax()],
            'pvar':pvar,
        }

build = Dev
