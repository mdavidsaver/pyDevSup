#!./bin/linux-x86-debug/devsup

epicsEnvSet("PYTHONPATH", "${PWD}/python:${PWD}/testApp")

dbLoadDatabase("dbd/devsup.dbd")
devsup_registerRecordDeviceDriver(pdbbase)

#py "import devsup.hooks"
#py "devsup.hooks.debugHooks()"

py "import test2"
py "test2.addDrv('AAAA')"
py "test2.addDrv('BBBB')"

dbLoadRecords("db/test.db","P=md:")

iocInit()
