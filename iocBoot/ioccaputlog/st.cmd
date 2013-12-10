#!../../bin/linux-x86_64/softIocPy

< envPaths

dbLoadDatabase("../../dbd/softIocPy.dbd",0,0)
softIocPy_registerRecordDeviceDriver(pdbbase) 

dbLoadRecords("../../db/logwatch.db","N=ACC-CT{}Log-I,FNAME=/var/log/epics/epics.log,FILTER=logwatch.caputlog")

iocInit()

dbl > records.dbl
