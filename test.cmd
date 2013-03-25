#!./bin/linux-x86-debug/devsup

epicsEnvSet("PYTHONPATH", "${PWD}/python")

dbLoadDatabase("dbd/devsup.dbd")
devsup_registerRecordDeviceDriver(pdbbase)

evalPy "import sys"
evalPy "print sys.path"
evalPy "import devsup.hooks"
evalPy "devsup.hooks.debugHooks()"
evalPy "import devsup.db"

dbLoadRecords("test.db","")

iocInit

evalPy "print devsup.db.Record('test:rec')"
evalPy "print devsup.db.Record('does:not:exist')"
