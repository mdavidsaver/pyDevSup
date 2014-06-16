#!../../bin/linux-x86/softIocPy2.6

< envPaths

dbLoadRecords("../../db/logwatch.db","N=ACC-CT{}Log-I,FNAME=/var/log/epics/epics.log,FILTER=logwatch.caputlog")

iocInit()

dbl > records.dbl
