# -*- coding: utf-8 -*-
from __future__ import print_function

import os.path, errno, time, sys

import numpy as np

import inotifyy as inot

from devsup.hooks import addHook
from devsup.util import importmod, StoppableThread
from devsup.db import IOScanListThread

py3 = sys.version_info[0]>=3

mask=inot.IN_CREATE|inot.IN_DELETE|inot.IN_MOVED_FROM|inot.IN_MODIFY

class LogWatcher(StoppableThread):

    def __init__(self, rec, args):
        super(LogWatcher, self).__init__()
        self.fname = args
        dir, self.fpart = os.path.split(self.fname)
        if not os.path.isdir(dir):
            raise RuntimeError("Directory '%s' does not exist"%dir)

        self.IN = inot.INotify()
        self.dirwatch = self.IN.add(self.event, dir, mask)

        self.scan = IOScanListThread()
        self.allowScan = self.scan.add

        addHook('AfterIocRunning', self.start)
        addHook('AtIocExit', self.join)

        self.arr = rec.field('VAL').getarray()
        self.fd = None

        filt = rec.info('logfilter',"")
        if filt:
            filt = importmod(filt)
            filt = filt.filter(self.fname)
        self.filt = filt

        print(rec, 'will watch', self.fname)
        print(rec, 'filtered with',self.filt)

    def detach(self, rec):
        pass

    def process(self, rec, reason=None):
        if reason is None:
            return
        ts, reason = reason
        if py3:
            reason = reason.encode('ascii')
        buf = np.frombuffer(reason, dtype=self.arr.dtype)
        buf = buf[:rec.NELM-1]
        self.arr[:buf.size] = buf
        self.arr[buf.size] = 0
        rec.NORD = buf.size+1
        
        if ts:
            rec.setTime(ts)
        else:
            rec.setTime(time.time())

    def join(self):
        print("Stopping logger for",self.fname)
        self.IN.close()
        print("Waiting for",self.fname)
        ret = super(LogWatcher, self).join()
        print("Joined",self.fname)
        return ret

    def run(self):
        print("log watcher staring",self.fname)
        self.log("Starting")
        self.openfile()
        self.catfile()
        self.IN.loop()

    def event(self, evt, mask, cookie, path):
        if path!=self.fpart:
            return

        if mask&inot.IN_CREATE:
            self.log('Log file created')
            self.openfile()

        if mask&(inot.IN_DELETE|inot.IN_MOVED_FROM):
            self.log("Log file deleted/renamed")
            self.closefile()

        if mask&(inot.IN_MODIFY):
            self.catfile()

    def openfile(self):
        self.closefile()
        try:
            self.fd = open(self.fname, 'r')
            self.pos = self.fd.tell()
        except IOError as e:
            if e.errno==errno.ENOENT:
                return
            raise

    def closefile(self):
        if self.fd:
            self.fd.close()
        self.fd, self.buf = None, None

    def catfile(self):
        if not self.fd:
            return
        self.fd.seek(0,2) # Seek end
        end = self.fd.tell()
        if end < self.pos:
            self.log("File size decreased from %d to %d, assuming truncation"%(self.pos,end))
            self.pos, self.buf = 0, None
        self.fd.seek(self.pos)

        for L in self.fd.readlines():
            if L[-1]!='\n':
                if self.buf:
                    self.buf += L
                else:
                    self.buf = L
                break
            elif self.buf:
                L, self.buf = self.buf+L, None
            self.log(L[:-1]) # Skip newline

        self.pos = self.fd.tell()

    def log(self, msg):
        ts = None
        if self.filt:
            ts, msg = self.filt.apply(msg)
        self.scan.interrupt(reason=(ts, msg))

build = LogWatcher
