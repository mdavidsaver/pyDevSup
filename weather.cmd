#!./bin/linux-x86/softIocPy

epicsEnvSet("PYTHONPATH", "${PWD}/python")

dbLoadDatabase("dbd/softIocPy.dbd")
softIocPy_registerRecordDeviceDriver(pdbbase)

dbLoadRecords("db/weather.db","P=kisp:,LOC=KISP")
#dbLoadRecords("db/weather.db","P=khwv:,LOC=KHWV")
#dbLoadRecords("db/weather.db","P=unnt:,LOC=UNNT")

iocInit()

dbl > weather.dbl
