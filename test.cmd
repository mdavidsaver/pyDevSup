#!./bin/linux-x86/softIocPy

epicsEnvSet("PYTHONPATH", "${PWD}/python/linux-x86:${PWD}/testApp")

dbLoadDatabase("dbd/softIocPy.dbd")
softIocPy_registerRecordDeviceDriver(pdbbase)

#py "import devsup.hooks"
#py "devsup.hooks.debugHooks()"

py "import test2"
py "test2.addDrv('AAAA')"
py "test2.addDrv('BBBB')"

dbLoadRecords("db/test.db","P=md:")

iocInit()
