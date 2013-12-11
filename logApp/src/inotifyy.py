from __future__ import print_function

# Pull in the IN_* macros
from _inotifyy import *

import _inotifyy
import select, os, errno

_flags = {}
for k in dir(_inotifyy):
    if k.startswith('IN_') and k!='IN_ALL_EVENTS':
        _flags[k[3:]] = getattr(_inotifyy, k)
del k

def decodeMask(mask):
    ret = []
    for k,v in _flags.iteritems():
        if mask&v:
            ret.append(k)
    return '%s %s'%(hex(mask),ret)

class IToken(object):
    def __init__(self, cb, path, inot, wd):
        self._cb, self._inot, self._wd = cb, inot, wd
        self.path = path
    def __del__(self):
        self.close()
    def close(self):
        if self._inot:
            self._inot._del(self._wd)
            self._inot = None
    def __str__(self):
        return 'IToken(%d,"%s")'%(self._wd, self.path)
    __repr__=__str__

class INotify(_inotifyy.INotify):
    def __init__(self):
        self.__done = False
        self.__listen, self.__wake = os.pipe()
        self.__wds = {}
        super(INotify, self).__init__()

    def close(self):
        self.__done = True
        os.write(self.__wake,b'*')

    def add(self, callback, path, mask=IN_ALL_EVENTS):
        wd = super(INotify, self).add(path, mask)
        try:
            return self.__wds[wd]
        except KeyError:
            tok = IToken(callback, path, self, wd)
            self.__wds[wd] = tok
            return tok

    def loop(self):
        while not self.__done:
            rds, _wts, _exs = select.select([self.fd, self.__listen], [], [])
            if self.fd in rds:
                for wd, mask, cookie, path in self.read():
                    try:
                        tok = self.__wds[wd]
                    except KeyError:
                        pass # ignore unknown wd
                    else:
                        tok._cb(tok, mask, cookie, path)
            elif self.__listen in rds:
                os.read(self.__listen,1024)
        self.__done = False



def cmdlisten(files):
    print("Listening for",*files)
    if len(files)==0:
        return

    def event(evt, mask, cookie, path):
        print('Event',evt, decodeMask(mask), cookie, path)

    IN = INotify()

    wds = [IN.add(event, P) for P in files]
    print(wds)

    return IN

class cmdtail(object):
    def __init__(self, fname):
        import os.path
        self.fname = fname
        dirname, self.fpart= os.path.split(fname)
        self.IN = INotify()
        self.loop = self.IN.loop

        mask=IN_CREATE|IN_DELETE|IN_MOVED_FROM|IN_MODIFY

        self.dirwd = self.IN.add(self.direvt, dirname, mask)
        self.fd = None
        self.startfile()
        self.catfile()

    def startfile(self):
        self.closefile()
        try:
            self.fd = open(self.fname, 'r')
        except IOError as e:
            if e.errno==errno.ENOENT:
                print(self.fname, "Doesn't exist yet")
                return
        #self.catfile()

    def closefile(self):
        if self.fd:
            print("Closing previous")
            self.fd.close()
        self.fd = None

    def catfile(self):
        if not self.fd:
           return
        op = self.fd.tell()
        self.fd.seek(0, 2)
        end = self.fd.tell()
        if end<op:
            print(self.fname,'got shorter...  Assuming truncation')
            self.fd.seek(0, 0)
        else:
            self.fd.seek(op, 0)
        while True:
            D = self.fd.read(1024)
            if D:
                print(D)
            else:
                break

    def direvt(self, evt, mask, cookie, path):
        if path!=self.fpart:
            return
        print('Dir event',evt, decodeMask(mask), cookie, path)
        if mask&IN_CREATE:
            print(self.fname,'appears')
            self.startfile()
        if mask&IN_DELETE:
            print(self.fname,'is deleted')
            self.closefile()
        if mask&IN_MOVED_FROM:
            print(self.fname,'is renamed')
            self.closefile()
        if mask&IN_MODIFY:
            print(self.fname,'is modified')
            self.catfile()

if __name__=='__main__':
    import sys
    if len(sys.argv)<=2:
        print("Usage: inotifyy listen <path1> [path2] ...")
        print("    or inotifyy tail <path>")
        sys.exit(1)
    elif sys.argv[1]=='listen':
        IN = cmdlisten(sys.argv[2:])
    elif sys.argv[1]=='tail':
        IN = cmdtail(sys.argv[2])
    IN.loop()
