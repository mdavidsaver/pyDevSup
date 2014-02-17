#!../../bin/linux-x86_64/softIocPy2.6

< envPaths

dbLoadDatabase("../../dbd/softIocPy.dbd",0,0)
softIocPy_registerRecordDeviceDriver(pdbbase)

py "import logging"
py "logging.basicConfig(level=logging.INFO)"

epicsEnvSet("BASE","/var/cache/channelarchiver")
epicsEnvSet("PPAT","The original process ID was ([0-9]+)")

dbLoadRecords("../../db/pidmon.db","N=ACC-CT{Bck}General-I,SCAN=10 second,FILE=$(BASE)/general/archive_active.lck,PAT=$(PAT)")

iocInit()

dbl > records.dbl
