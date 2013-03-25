try:
    import _dbapi
except ImportError:
    import _nullapi as _dbapi
from _dbapi import _hooks, _hooktable

__all__ = [
    "hooknames",
    "addHook",
    "debugHooks",
]

hooknames = _hooks.keys()

def addHook(state, func):
    """addHook("stats", funcion)
    Add callback function to IOC start sequence.
    
    def show():
        print 'State Occurred'
    addHook("AfterIocRunning", show)
    """
    sid = _hooks[state]
    try:
        slist = _hooktable[sid]
    except KeyError:
        slist = []
        _hooktable[sid] = slist

    slist.append(func)


def debugHooks():
    """Install debugging print to hooks
    """
    for h in hooknames:
        def _showstate(state=h):
            print 'Reached state',state
        addHook(h, _showstate)
