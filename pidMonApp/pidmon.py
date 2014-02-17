# -*- coding: utf-8 -*-

import logging
LOG = logging.getLogger(__name__)

import re, os, errno

from devsup import MAJOR_ALARM, READ_ALARM

class PIDMon(object):
    def __init__(self, rec, lnk):
        self.fname = rec.info('pidfile')
        pat = rec.info('pidpat',None)
        if not pat:
            pat = '([1-9]+)'
        LOG.info('%s: in %s find "%s"', rec.NAME, self.fname, pat)
        self.pat = re.compile(pat)

    def detach(self, rec):
        pass
    def allowScan(self, rec):
        return False
    def process(self, rec, reason=None):
        try:
            ok, pid = False, None

            LOG.debug('Open %s', self.fname)
            with open(self.fname, 'r') as F:
                for line in map(str.rstrip, F.readlines()):
                    LOG.debug('Read: %s', line)
                    M = self.pat.match(line)
                    if M:
                        LOG.debug('Match: %s', M.groups())
                        pid = int(M.group(1))
                        break

            if pid is None:
                rec.VAL = 'no PID in PID file'
                return

            LOG.debug('Testing PID %d', pid)
            os.kill(pid, 0) # 0 doesn't signal, but does check for existance

            rec.VAL = 'Running'
            ok = True

        except IOError as e:
            if e.errno==errno.ENOENT:
                rec.VAL = 'No PID file'
            else:
                rec.VAL = str(e)

        except OSError as e:
            if e.errno==errno.ESRCH:
                rec.VAL = 'Process not running'
            else:
                rec.VAL = str(e)

        finally:
            if not ok:
                rec.setSevr(MAJOR_ALARM, READ_ALARM)

build = PIDMon
