ptable Package
==============

.. module:: devsup.ptable

The :py:mod:`devsup.ptable` module provides the means to define a Parameter Table,
which is something like a dictionary (parameter name <-> dict key)
where a parameter may be associated (attached) with zero or more EPICS records.

Changes to a parameter may be reflected in the attached records.
A change in an attached record will update the parameter value,
and optionally, involve some functions (actions) to make some use of the new value.

The basis of all tables is the :py:class:`TableBase` class.  User code
will typically sub-class :py:class:`TableBase`.

Defining Parameters
^^^^^^^^^^^^^^^^^^^

.. autoclass:: Parameter
    :members:

.. autoclass:: ParameterGroup
    :members:

.. autoclass:: TableBase
    :members:

Runtime Access
^^^^^^^^^^^^^^

.. autoclass :: _ParamInstance
    :members:

.. autoclass :: _ParamGroupInstance
    :members:

Device Support
^^^^^^^^^^^^^^

A general purpose device support is provided to access table parameters.
The input/output link format is "@devsup.ptable <tablename> <param> [optional]"
