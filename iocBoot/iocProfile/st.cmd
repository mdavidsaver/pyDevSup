#!../../bin/linux-x86_64/softIocPy3.7

epicsEnvSet(PREFIX, "LCam1:")
epicsEnvSet(STATS , "Stats1:")
epicsEnvSet(FIT   , "Fit1")
epicsEnvSet("EPICS_CA_MAX_ARRAY_BYTES","100000")

py "import logging"
py "logging.basicConfig(level=logging.INFO)"

dbLoadRecords("profile.db", "P=$(PREFIX)$(FIT)X:,INP=$(PREFIX)$(STATS)ProfileAverageX_RBV,MAXPIXEL=255")
dbLoadRecords("profile.db", "P=$(PREFIX)$(FIT)Y:,INP=$(PREFIX)$(STATS)ProfileAverageY_RBV,MAXPIXEL=255")

#dbLoadRecords("../../db/iocAdminSoft.db", "IOC=SR-CT{IOC:FPM}")
#dbLoadRecords("../../db/save_restoreStatus.db", "P=SR-CT{IOC:FPM}")
#save_restoreSet_status_prefix("SR-CT{IOC:FPM}")

#set_savefile_path("${PWD}/as","/save")
#set_requestfile_path("${PWD}/as","/req")

#set_pass0_restoreFile("ioc_settings.sav")

iocInit()

#makeAutosaveFileFromDbInfo("as/req/ioc_settings.req", "autosaveFields_pass0")
#create_monitor_set("ioc_settings.req", 10, "")
