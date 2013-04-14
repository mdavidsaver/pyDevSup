from __future__ import print_function

import traceback
from collections import defaultdict

try:
    import _dbapi
except ImportError:
    import devsup._nullapi as _dbapi

__all__ = [
    "hooknames",
    "addHook",
    "debugHooks",
]

hooknames = _dbapi._hooks.keys()

_revnames = dict([(v,k) for k,v in _dbapi._hooks.iteritems()])

_hooktable = defaultdict(list)

def addHook(state, func):
    """addHook("stats", funcion)
    Add callable to IOC start sequence.
    
    Callables are run in the order in
    which they were added (except for 'AtIocExit').
    
    >>> def show():
    ...     print 'State Occurred'
    >>> addHook("AfterIocRunning", show)
    
    An additional special hook 'AtIocExit' may be used
    for cleanup actions during IOC shutdown.
    """
    sid = _dbapi._hooks[state]
    _hooktable[sid].append(func)


def debugHooks():
    """Install debugging print to hooks
    """
    for h in hooknames:
        def _showstate(state=h):
            print('Reached state',state)
        addHook(h, _showstate)

def _runhook(sid):
    name = _revnames[sid]
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
