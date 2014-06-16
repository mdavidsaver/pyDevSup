#!../../bin/linux-x86/softIocPy2.7

< envPaths

py "import logging"
py "logging.basicConfig(level=logging.INFO)"

epicsEnvSet("BASE","/var/cache/channelarchiver")
epicsEnvSet("PPAT","The original process ID was ([0-9]+)")

dbLoadRecords("../../db/pidmon.db","N=ACC-CT{Bck}Diag-I,SCAN=10 second,FILE=$(BASE)/diag/archive_active.lck,PAT=$(PPAT)")
dbLoadRecords("../../db/pidmon.db","N=ACC-CT{Bck}Gen-I,SCAN=10 second,FILE=$(BASE)/general/archive_active.lck,PAT=$(PPAT)")
dbLoadRecords("../../db/pidmon.db","N=ACC-CT{Bck}InjVA-I,SCAN=10 second,FILE=$(BASE)/inj-va/archive_active.lck,PAT=$(PPAT)")
dbLoadRecords("../../db/pidmon.db","N=ACC-CT{Bck}PSCrit-I,SCAN=10 second,FILE=$(BASE)/PS_crit/archive_active.lck,PAT=$(PPAT)")
dbLoadRecords("../../db/pidmon.db","N=ACC-CT{Bck}RadMon-I,SCAN=10 second,FILE=$(BASE)/RadMon/archive_active.lck,PAT=$(PPAT)")
dbLoadRecords("../../db/pidmon.db","N=ACC-CT{Bck}Skids-I,SCAN=10 second,FILE=$(BASE)/waterskid/archive_active.lck,PAT=$(PPAT)")

iocInit()

dbl > records.dbl
