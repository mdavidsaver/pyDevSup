#!./bin/linux-x86/softIocPy

dbLoadDatabase("dbd/softIocPy.dbd")
softIocPy_registerRecordDeviceDriver(pdbbase)

py "import devsup; print devsup.HAVE_DBAPI"
py "import sys; sys.path.insert(0,'${PWD}/testApp')"
py "print sys.path"

#py "import devsup.hooks"
#py "devsup.hooks.debugHooks()"

py "import test2"
py "test2.addDrv('AAAA')"
py "test2.addDrv('BBBB')"

dbLoadRecords("db/test.db","P=md:")

iocInit()
