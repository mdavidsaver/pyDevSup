#!../../bin/linux-x86/softIocPy2.6

< envPaths

cd("$(TOP)")

epicsEnvSet("http_proxy", "http://proxy:8888/")

epicsEnvSet("PYTHONPATH", "${TOP}/python/$(ARCH)")

dbLoadRecords("db/weather.db","P=CF:Ext{KISP},LOC=KISP")
dbLoadRecords("db/weather.db","P=CF:Ext{KHWV},LOC=KHWV")
dbLoadRecords("db/weather.db","P=CF:Ext{UNNT},LOC=UNNT")

iocInit()

dbl > weather.dbl
