from __future__ import print_function

import traceback
from functools import wraps
from collections import defaultdict

from . import _dbapi

__all__ = [
    "hooknames",
    "addHook",
    "initHook",
    "debugHooks",
]

hooknames = list(_dbapi._hooks.keys())

_revnames = dict([(v,k) for k,v in _dbapi._hooks.items()])

_hooktable = defaultdict(list)

def addHook(state, func):
    """addHook("stats", funcion)
    Add callable to IOC start sequence.
    
    Callables are run in the order in
    which they were added (except for 'AtIocExit').
    
    >>> def show():
    ...     print('State Occurred')
    >>> addHook("AfterIocRunning", show)
    
    An additional special hook 'AtIocExit' may be used
    for cleanup actions during IOC shutdown.
    """
    sid = _dbapi._hooks[state]
    _hooktable[sid].append(func)


def initHook(state):
    """Decorator for initHook functions

    @initHook("AfterIocRunning")
    def myfn():
        pass
    """
    def _add(fn):
        addHook(state, fn)
        return fn
    return _add

def debugHooks():
    """Install debugging print to hooks
    """
    for h in hooknames:
        def _showstate(state=h):
            print('Reached state',state)
        addHook(h, _showstate)

def _runhook(sid):
    name = _revnames.get(sid) or 'initHook%d'%sid
    pop = -1 if name=='AtIocExit' else 0
    fns = _hooktable.get(sid)
    if fns is not None:
        while len(fns)>0:
            fn = fns.pop(pop)
            try:
                fn()
            except:
                print("Error running",name,"hook",fn)
                traceback.print_exc()
