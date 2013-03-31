#!./bin/linux-x86-debug/devsup

epicsEnvSet("PYTHONPATH", "${PWD}/python:${PWD}/testApp")

dbLoadDatabase("dbd/devsup.dbd")
devsup_registerRecordDeviceDriver(pdbbase)

#evalPy "import devsup.hooks"
#evalPy "devsup.hooks.debugHooks()"

evalPy "import test2"
evalPy "test2.addDrv('AAAA')"
evalPy "test2.addDrv('BBBB')"

dbLoadRecords("db/test.db","P=md:")

iocInit()
