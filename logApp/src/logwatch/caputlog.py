"""
linacioc01.cs.nsls2.local:39907 Mon Dec  9 07:07:53 2013 09-Dec-13 07:07:48 diagioc-spc softioc LN-DG{SCR:6}In-Cmd.VAL new=0 old=0

Input format is
 iochost:port DoW Mon day H:M:S Year D-M-Y H:M:S userhost user PV msg
"""

import re, time

R = re.compile(r"""
# iochost:port
 (\S+)\s
# date 1 (in logger host TZ)
 (?P<ts>\S+\s\S+\s+\S+\s\S+\s\S+)\s
# date 2 (in IOC TZ)
 (\S+\s\S+)\s
# userhost
 (?P<host>\S+)\s
# username
 (?P<user>\S+)\s
# PV
 (?P<pv>\S+)\s
# message
 (?P<msg>\S.*)
""", re.VERBOSE)

class CAPutLogFilter(object):
    def __init__(self, fname):
        self.fname = fname
        self.noise = False

    def __str__(self):
        return '%s(%s)'%(str(type(self)), self.fname)
    __repr__ = __str__

    def apply(self, line):
        M = R.match(line)
        if not M:
            # lines not matching the caputlog format
            # are passed through verbatim
            return None, line
        D = M.groupdict()
        try:
            ts = time.mktime(time.strptime(D['ts'],'%a %b %d %H:%M:%S %Y'))
            self.noise = False
        except ValueError:
            if not self.noise:
                print("Failed to parse time",D['ts'])
                self.noise = True
            ts = None
            raise
        msg = "%(user)s %(host)s    %(pv)s %(msg)s"%D
        return ts, msg

filter = CAPutLogFilter
