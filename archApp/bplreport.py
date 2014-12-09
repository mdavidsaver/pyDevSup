# -*- coding: utf-8 -*-

import time, sched, urllib2, json
from devsup.db import IOScanListBlock
from devsup.hooks import initHook
from devsup.util import StoppableThread

class BPLReport(object):
    reports = {}

    def __init__(self, name, url, period):
        self.name = name
        self.url, self.period = url, period
        self.result = None
        self.reports[name] = self
        self.scan = IOScanListBlock()

    def fail(self):
        self.result = None

    def process(self):
        self.result = None
        R = urllib2.urlopen(self.url, timeout=3)
        try:
            if R.getcode()!=200:
                print 'Fail',R.getcode(), self.url
                self.result = None
                return
            self.result = json.load(R)
        except:
            print 'Error fetching',self.url
            import traceback
            traceback.print_exc()
        finally:
            R.close()
            self.result_time = time.time()
        self.scan.interrupt(reason = self.result)

add = BPLReport

class ReportRunner(StoppableThread):
    class _Done(Exception):
        pass

    def _sleep(self, time):
        if not self.sleep(time):
            raise self._Done()

    def _proc(self, R):
        self._S.enter(R.period, 0, self._proc, (R,))
        try:
            R.process()
        except:
            print 'Error in processing',R.url
            import traceback
            traceback.print_exc()
            R.fail()

    def run(self):
        self._S = S = sched.scheduler(time.time, self._sleep)

        for R in BPLReport.reports.itervalues():
            S.enter(0, 0, self._proc, (R,))

        try:
            S.run()
        except self._Done:
            print 'BPL worker exit'
        except:
            print 'Error in scheduler'
            import traceback
            traceback.print_exc()

_worker = ReportRunner()

@initHook("AfterIocRunning")
def _startWorker():
    _worker.start()
    print 'BPL worker started'

@initHook("AtIocExit")
def _stopWorker():
    print 'BPL worker stopping'
    _worker.join()
    print 'BPL worker stopped'

class ReportItem(object):
    raw = True
    def __init__(self, rec, args):
        # "<operation> <report>.<index>.<attribute> "
        opname, src = args.split(None,2)[:2]
        self.report, self.idx, self.attrib = src.split('.',2)
        self.idx = int(self.idx)
        self.R = BPLReport.reports[self.report]

        self.allowScan = self.R.scan.add
        self.process = getattr(self, 'process_'+opname)

    def detach(self, rec):
        pass

    def process_fetch_float(self, rec, reason=None):
        R = self.R.result
        invalid = True
        if R is not None and len(R)>self.idx:
            try:
                rec.VAL = float(str(R[self.idx][self.attrib]).translate(None,','))
            except KeyError:
                pass
            else:
                invalid = False

        rec.UDF = invalid
        rec.setTime(self.R.result_time)

    def process_fetch_int(self, rec, reason=None):
        R = self.R.result
        invalid = True
        if R is not None and len(R)>self.idx:
            try:
                rec.VAL = int(R[self.idx][self.attrib])
            except KeyError:
                pass
            else:
                invalid = False

        rec.UDF = invalid
        rec.setTime(self.R.result_time)

    def process_fetch_string(self, rec, reason=None):
        R = self.R.result
        invalid = True
        if R is not None and len(R)>self.idx:
            try:
                rec.VAL = R[self.idx][self.attrib].encode('ascii')
            except KeyError:
                pass
            else:
                invalid = False

        if invalid:
            rec.setSevr() # default is INVALID_ALARM
        rec.setTime(self.R.result_time)

    def process_fetch_length(self, rec, reason=None):
        if self.R.result is not None:
            rec.VAL = len(self.R.result)
        rec.UDF = self.R.result is None
        rec.setTime(self.R.result_time)

build = ReportItem
