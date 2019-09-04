# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
LOG = logging.getLogger(__name__)

import threading, inspect

_tables = {}

from .db import IOScanListThread
from . import INVALID_ALARM, UDF_ALARM

__all__ = [
    'Parameter',
    'ParameterGroup',
    'TableBase',
    'build'
]

# Reason code to cause a record to read a new value from a table parameter
_INTERNAL = object()

# action types
# Default action, call whenever a record associated with the parameter
# is processed
_ONPROC = lambda n,o:True
# Run action when the value of a parameter changes
_ONCHANGE = lambda n,o:n!=o
_ISVALID = lambda n,o:n is not None
_ISNOTVALID = lambda n,o:n is None

def _add_action(self, act, fn):
    try:
        L = fn._ptable_action
    except AttributeError:
        L = fn._ptable_action = []
    L.append((self,act,fn))
    return fn

class Parameter(object):
    """Define a parameter in a table.

    When a sub-class of TableBase is instantiated, parameters become
    py:class:`_ParamInstance` instances.

    >>> class MyTable(TableBase):
        A = Parameter()
        B = Parameter(name='bb')
        C = Parameter(iointr=True)
    >>>
    
    Defines a table with three parameters.  The second 'B' defines a different
    parameter name and attribute name.  The parameter name 'bb' will be used
    in .db files, which self.B will be used for access from the table instance.
    
    When _iointr_ is True, then attached device support may use SCAN='I/O Intr',
    which is triggered with the method self.B.notify().
    
    This class has several methods which may be used decorate member functions
    as actions when the value of a parameter is (possibly) changed.
    """
    def __init__(self, name=None, iointr=False):
        self.name, self._iointr = name, iointr
    def onproc(self, fn):
        """Decorator run a member function action whenever
        an attached device support processes.
        
        >>> class MyTable(TableBase):
            A = Parameter()
            @A.onproc
            def action(self, oldval):
                print 'A changed from',oldval,'to',self.A.value
        """
        return _add_action(self, _ONPROC, fn)
    def onchange(self, fn):
        "Decorator to run an action when the value of a parameter is changed."
        return _add_action(self, _ONCHANGE, fn)
    def isvalid(self, fn):
        "Decorator to run an action when the value is valid"
        return _add_action(self, _ISVALID, fn)
    def isnotvalid(self, fn):
        "Decorator to run an action when the value is *not* valid"
        return _add_action(self, _ISNOTVALID, fn)
    def oncondition(self, cond):
        """Decorator which allows a custom condition function to be specified.
        
        This function will be invoked with two argument cond(newval,oldval)
        and is expected to retur a bool.
        
        >>> class MyTable(TableBase):
            A = Parameter()
            @A.oncondition(lambda n,o:n<5)
            def action(self, oldval):
                print self.A.value,'is less than 5'
        """
        def decorate(fn, cond=cond, self=self):
            return _add_action(self, cond, fn)
        return decorate

class ParameterGroup(object):
    """A helper for defining actions on groups of parameters

    When a sub-class of TableBase is instantiated, parameter groups become
    py:class:`_ParamGroupInstance` instances.

    >>> class MyTable(TableBase):
        A = Parameter()
        B = Parameter(name='bb')
        grp = ParameterGroup([A,B])
    >>>

    This class has several methods which may be used decorate member functions
    as actions based on the value of parameters in this group.
    """
    def __init__(self, params, name=None):
        self.params, self.name = params, name
    def onproc(self, fn):
        """Decorator run a member function action whenever
        a device support attached to any parameter in the group processes.
        
        >>> class MyTable(TableBase):
            A, B = Parameter(), Parameter()
            grp = ParameterGroup([A,B])    
            @grp.onproc
            def action(self):
                print self.A.value, self.B.value
        """
        return _add_action(self, _ONPROC, fn)
    def allvalid(self, fn):
        "Decorator to run an action when all parameters have valid values"
        return _add_action(self, (all, lambda p:p.isvalid), fn)
    def anynotvalid(self, fn):
        "Decorator to run an action when any parameters has an invalid value"
        return _add_action(self, (any, lambda p:not p.isvalid), fn)
    def oncondition(self, fmap, freduce=all):
        """Decorator for a custom condition.
        
        The condition is specified in two parts, a map function, and a reduce function.
        The map function is applied to each parameter in the group.  Then a list
        of the results is passed to the reduce function.  If not specified,
        the default reducing function is all (map func must return bool).
        
        >>> class MyTable(TableBase):
            A, B = Parameter(), Parameter()
            grp = ParameterGroup([A,B])    
            @grp.oncondition(lambda v:v>5, any)
            def action(self):
                # either A or B is greater than 5
                print self.A.value, self.B.value
        """
        def decorate(fn, fmap=fmap, freduce=freduce, self=self):
            return _add_action(self, (freduce, fmap), fn)
        return decorate

class _ParamInstance(object):
    """Access to a parameter at runtime.
    """
    def __init__(self, table, name, scan):
        self.name = name
        self.table, self.scan, self._value = table, scan, None
        self.alarm, self.actions = 0, []
        self._groups = set()
    def _get_value(self):
        return self._value
    def _set_value(self, val):
        self._value = val
        self.alarm = 3 if val is None else 0
    value = property(_get_value, _set_value, doc="The current parameter value")
    @property
    def isvalid(self):
        """Is the parameter value valid (not None and no INVALID_ALARM)
        """
        return self.alarm < INVALID_ALARM and self._value is not None
    def notify(self):
        """Notify attached records of parameter value change.
        A no-op unless Parameter(iointr=True)
        """
        if self.scan:
            self.scan.interrupt(_INTERNAL)
    def addAction(self, fn, cond=None):
        """Add an arbitrary action at runtime
        """
        self.actions.append((cond, fn))
    def _exec(self, oldval=None):
        for C, F in self.actions:
            if C(self.value, oldval):
                F()

class _ParamGroupInstance(object):
    """Runtime access to a group of parameters.
    """
    def __init__(self, table, name):
        self.table, self.name = table, name
        self.actions, self._params = [], None
    def __iter__(self):
        "Iterate over all parameter instances in this group"
        return iter(self._params)
    def addAction(self, fn, fmap, freduce=all):
        "Add an arbitrary action at runtime"
        self.actions.append((fmap, freduce, fn))
    def _exec(self):
        for M, R, F in self.actions:
            if R(map(M, self._params)):
                F()
    def allValid(self):
        "Quick test if all parameters are valid"
        for P in self._params:
            if not P.isvalid:
                return False
        return True

class _ParamSupBase(object):
    def __init__(self, inst, rec, info):
        self.inst, self.info = inst, info
        # Determine which field to use to store the value
        fname = rec.info('pyfield','VAL')
        self.raw = fname!='RVAL'
        self.vfld = rec.field(fname)
        self.vdata = None
        if len(self.vfld)>1:
            self.vdata = self.vfld.getarray()
    def detach(self, rec):
        pass
    def allowScan(self, rec):
        if self.inst.scan:
            return self.inst.scan.add(rec)

class _ParamSupGet(_ParamSupBase):
    def process(self, rec, reason=None):
        """Read a value from the table into the record
        """
        with self.inst.table.lock:
            nval, alrm = self.inst.value, self.inst.alarm
            self.inst.table.log.debug('%s -> %s (%s)', self.inst.name, rec.NAME, nval)

        if nval is not None:
            if self.vdata is None:
                self.vfld.putval(nval)
            else:
                if len(nval)>len(self.vdata):
                    nval = nval[:len(self.vdata)]
                self.vdata[:len(nval)] = nval
                self.vfld.putarraylen(len(nval))
            if alrm:
                rec.setSevr(alrm)
        else:
            # undefined value
            rec.setSevr(INVALID_ALARM, UDF_ALARM)

class _ParamSupSet(_ParamSupGet):
    def process(self, rec, reason=None):
        """Write a value from the record into the table
        """
        if reason is _INTERNAL:
            # sync table to record
            super(_ParamSupSet,self).process(rec,reason)

        else:
            # sync record to table
            self.inst.table.log.debug('%s <- %s (%s)', self.inst.name, rec.NAME, rec.VAL)
            if self.vdata is None:
                nval = self.vfld.getval()
            else:
                # A copy is made which can be used without locking the record
                nval = self.vdata[:self.vfld.getarraylen()].copy()

            with self.inst.table.lock:
                oval, self.inst.value = self.inst.value, nval
                
                # Execute actions
                self.inst._exec(oval)
                for G in self.inst._groups:
                    G._exec()

class TableBase(object):
    """Base class for all parameter tables.
    
    Sub-class this and populate with :py:class:`Parameter` and  :py:class:`ParameterGroup`.
    
    #When a table is instantiated it must be given a unique name.
    
    >>> class MyTable(TableBase):
        ...
    >>> x=MyTable(name='xyz')
    >>>
    """
    log = LOG
    ParamSupportGet = _ParamSupGet
    ParamSupportSet = _ParamSupSet
    def __init__(self, **kws):
        self.name = kws.pop('name')
        if self.name in _tables:
            raise KeyError("Table named '%s' already exists"%self.name)
        self.lock = threading.Lock()
        self._parameters = {}

        # Find Parameters and ParameterGroup in the class dictionary
        # and place appropriate things in the instance dictionary
        rparams = {}
        rgroups = {}
        for k,v in inspect.getmembers(self):
            if isinstance(v, Parameter):
                scan = None
                if not v.name:
                    v.name = k
                if v._iointr:
                    scan = IOScanListThread()
                P = _ParamInstance(self, v.name, scan)
                self._parameters[v.name] = P
                rparams[v] = P
                setattr(self, k, P)
            elif isinstance(v, ParameterGroup):
                if not v.name:
                    v.name = k
                G = _ParamGroupInstance(self, v.name)
                rgroups[v] = G
                setattr(self, k, G)

        # Populate groups with parameters
        for g,G in rgroups.items():
            ps = G._params = [rparams[v] for v in g.params]
            # reverse mapping from parameter to group(s)
            for P in ps:
                P._groups.add(G)

        # second pass to attach actions
        for k,v in inspect.getmembers(self):
            if hasattr(v, '_ptable_action'):
                for src,cond,cmeth in v._ptable_action:
                    # src is instance parameter or group
                    # cond is a callable for parameters, or a tuple for groups
                    # cmeth is the unbound method which is the action to take
                    if isinstance(src, Parameter):
                        P = rparams[src]
                        P.addAction(cmeth.__get__(self), cond)
                    elif isinstance(src, ParameterGroup):
                        G = rgroups[src]
                        G.addAction(cmeth.__get__(self), cond[1], cond[0])

        super(TableBase, self).__init__(**kws)
        self.log.info("Initialized ptable '%s'",self.name)
        _tables[self.name] = self

def build(rec, args):
    """Device support pattern
    
    "@devsup.ptable <tablename> set|get <param> [optional]"
    """
    parts = args.split(None,3)
    table, param = parts[0], parts[2]
    info = None if len(parts)<4 else parts[3]
    T = _tables[table]
    P = T._parameters[param]
    if parts[1]=='set':
        return T.ParamSupportSet(P, rec, info)
    elif parts[1]=='get':
        return T.ParamSupportGet(P, rec, info)
    else:
        raise ValueError("Not set or get")
    T.log.debug("Attaching ptable '%s, %s' to %s", T.name, P.name, rec.NAME)
    return T.ParamSupport(P, rec, info)
