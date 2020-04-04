# -*- coding: utf-8 -*-
"""
Gaussian fitting, intended for H/V image projection/profile fitting
"""

import logging
_log = logging.getLogger(__name__)

import time

import numpy
# fyi. https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#example-of-solving-a-fitting-problem
from scipy.optimize import least_squares

from devsup import MINOR_ALARM, MAJOR_ALARM, CALC_ALARM
from devsup.util import Worker
from devsup.hooks import initHook
from devsup.dset import AsyncOffload

def curve(F, X):
    offset, amp, mean, stddev = F
    return offset + amp * numpy.exp((-(X-mean)**2)/(2*stddev**2))

def error(F, X, Y):
    return curve(F, X) - Y

class Dev(AsyncOffload):
    inputs = {
        'A':'profile',
        'B':'imean',
        'C':'istddev',
        'D':'fullscale',
    }
    outputs = {
        'VALA':'fitted',
        'VALB':'mean',
        'VALC':'stddev',
        'VALD':'result', # 0 - Init, 1 - Success, 2 - Warning
        'VALE':'markerx',
        'VALF':'markery',
        'VALG':'time',
    }

    def inThread(self, profile, imean, istddev, fullscale):
        T0 = time.time()
        _log.debug('Process with %s elements', profile.shape)
        if len(profile)==0:
            return {'result':0}

        X = numpy.arange(len(profile)) # units of pixels

        # initial fit parameters (offset, amplitude, mean, stddev)
        F0 = [
            profile.min(),
            profile.max() - profile.min(),
            imean,
            istddev
        ]

        # upper bound for each parameter
        upper = [
            fullscale,
            fullscale,
            len(profile),
            len(profile)*0.75,
        ]

        # lower bound for each parameter
        lower = [
            0,
            10,
            0,
            4,
        ]

        _log.debug('Fitting with bounds lower=%s upper=%s', lower, upper)
        _log.debug('Initial fit parameters=%s', F0)

        res = least_squares(
            error,
            F0,
            bounds=(lower, upper),
            args=(X, profile),
        )
        _log.debug("Result %s %s %s", res.success, res.status, res.active_mask)

        Ff = res.x
        Yf = curve(Ff, X)

        ret = {
            'fitted':Yf,
            'mean':Ff[2],
            'stddev':Ff[3],
            'markerx':numpy.asarray([Ff[2], Ff[2]]),
            'markery':numpy.asarray([profile.min(), profile.max()]),
        }

        if not res.success:
            # fit fail
            ret['result'] = 0
            self.rec.setSevr(stat=CALC_ALARM, sevr=INVALID_ALARM)
        if res.status<=0 or (res.active_mask!=0).any():
            # no converge or param at limit
            ret['result'] = 2
            self.rec.setSevr(stat=CALC_ALARM, sevr=MINOR_ALARM)
        else:
            # OK
            ret['result'] = 1

        _log.debug('Fit %s parameters=%s', ret['result'], Ff)

        T1 = time.time()
        ret['time'] = (T1-T0)*1e3 # ms

        return ret

build = Dev
