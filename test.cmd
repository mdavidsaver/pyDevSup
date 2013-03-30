#!./bin/linux-x86-debug/devsup

epicsEnvSet("PYTHONPATH", "${PWD}/python:${PWD}/testApp")

dbLoadDatabase("dbd/devsup.dbd")
devsup_registerRecordDeviceDriver(pdbbase)

#evalPy "import devsup.hooks"
#evalPy "devsup.hooks.debugHooks()"

dbLoadRecords("db/test1.db","P=md:")

iocInit()
