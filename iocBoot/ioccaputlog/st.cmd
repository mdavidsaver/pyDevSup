#!../../bin/linux-x86_64/softIocPy

< envPaths

dbLoadDatabase("../../dbd/softIocPy.dbd",0,0)
softIocPy_registerRecordDeviceDriver(pdbbase) 

dbLoadRecords("../../db/logwatch.db","N=logrec,FNAME=/tmp/testlog")

iocInit()

dbl > records.dbl
