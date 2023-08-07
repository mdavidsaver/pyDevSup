# -*- coding: utf-8 -*-
from __future__ import print_function

import time

from weakref import WeakValueDictionary

from devsup.hooks import addHook
from devsup.util import StoppableThread
from devsup.db import IOScanListBlock

try:
  import pymetar
except ImportError:
  print("The pymetar package could not be imported!")
  raise

_isofmt = '%Y-%m-%d %H:%M:%SZ'

def iso2sec(str):
  """Convert the string returned by pymetar to posix time.
  """
  rtime = time.mktime(time.strptime(str, _isofmt))
  return rtime - (time.altzone if time.daylight else time.timezone)

_stations = WeakValueDictionary()

def getStation(name):
  try:
    return _stations[name]
  except KeyError:
    S = ReportScanner(name)
    _stations[name] = S
    return S

class DataWatcher(object):
  """Weather watcher device support.
  
  field(INP,"weather KISP getTemperatureCelsius")
  of
  field(INP,"weather KISP showID")
  """

  # disable automatic RVAL -> VAL conversion.
  # We will update VAL ourselves.
  raw = True

  def __init__(self, rec, args):
    station, name = args.split(None, 1)

    self.staid = station
    self.sta = getStation(station)
    self.attr = name
    self.last = None

    if name.startswith('get'):
      # Assume this is a method of pymetar.WeatherReport
      self.allowScan = self.sta.scan.add
      self.process = self.process_report
    else:
      # Our own internal info
      self.allowScan = self.sta.intscan.add
      self.process = getattr(self, name)
      try:
        rec.UDF = 0
      except AttributeError:
        pass

  def detach(self, rec):
    pass

  def showID(self, rec, report):
    rec.VAL = self.staid

  def updatePeriod(self, rec, report):
    rec.VAL = self.sta.updatePeriod/60.0

  def process_report(self, rec, report):
    if report is not None:
      self.last = report
    else:
      report = self.last

    if report is None:
      rec.setSevr()
      return

    fn = getattr(self.last, self.attr)
    newval = fn()
    try:
      rec.VAL = newval
    except ValueError:
      rec.setSevr()
    try:
      rec.UDF = 0
    except AttributeError:
      pass
    if rec.TSE==-2:
      rec.setTime(report._updatetime)

build = DataWatcher

class ReportScanner(StoppableThread):
  """Driver thread which occasionally polls for new metar data
  """

  def __init__(self, station):
    self.io = pymetar.ReportFetcher()
    self.station = station
    self.initPeriod = self.updatePeriod = 15*60.0
    self.minPeriod = 10*60.0
    self.maxPeriod = 2*60*60.0
    self.lastUpdate = None # Time of last report
    self.scan = IOScanListBlock() # I/O Intr scan for report data
    self.intscan = IOScanListBlock() # scan for internal info

    super(ReportScanner,self).__init__()

    addHook('AfterIocRunning', self.start)
    addHook('AtIocExit', self.join)

  def run(self):
    print("Starting scanner for ",self.station)

    while self.shouldRun():
      try:
        report = self.io.FetchReport(self.station)
        pymetar.ReportParser().ParseReport(report)

        rtime = iso2sec(report.getISOTime())
        report._updatetime = rtime
        #print('update',report.getISOTime(),rtime,self.lastUpdate)

        if self.lastUpdate is not None:
          if self.lastUpdate >= rtime:
            #print('No update, Wait a little longer next time')
            self.updatePeriod = self.initPeriod

          else: # self.lastUpdate < rtime
            #print('Got an update')
            self.updatePeriod = rtime - self.lastUpdate
            self.updatePeriod = max(self.minPeriod, min(self.updatePeriod, self.maxPeriod))
            self.updatePeriod += 15*60.0
            self.scan.interrupt(reason=report)
        else:
          self.scan.interrupt(reason=report)

        self.lastUpdate = rtime

      except Exception as e:
        print("download error for",self.station,":",e)
        self.updatePeriod = self.initPeriod
        from traceback import print_exc
        print_exc()

      self.intscan.interrupt()

      #print('Waiting',self.updatePeriod)
      self.sleep(self.updatePeriod)

    print('Done')
