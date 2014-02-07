#!./bin/linux-x86/softIocPy

dbLoadDatabase("dbd/softIocPy.dbd")
softIocPy_registerRecordDeviceDriver(pdbbase)

py "import logging"
py "logging.basicConfig(level=logging.DEBUG)"

py "import devsup; print devsup.HAVE_DBAPI"
py "import sys; sys.path.insert(0,'${PWD}/testApp')"
py "print sys.path"

#py "import devsup.hooks"
#py "devsup.hooks.debugHooks()"

py "import test2"
py "test2.addDrv('AAAA')"
py "test2.addDrv('BBBB')"

py "import test6"
py "test6.SumTable(name='tsum')"

dbLoadRecords("db/test.db","P=md:")
dbLoadRecords("db/test6.db","P=tst:,TNAME=tsum")

iocInit()
