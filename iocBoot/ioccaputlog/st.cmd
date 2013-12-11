#!../../bin/linux-x86/softIocPy2.6

< envPaths

dbLoadDatabase("../../dbd/softIocPy.dbd",0,0)
softIocPy_registerRecordDeviceDriver(pdbbase) 

dbLoadRecords("../../db/logwatch.db","N=ACC-CT{}Log-I,FNAME=/var/log/epics/epics.log,FILTER=logwatch.caputlog")

iocInit()

dbl > records.dbl
