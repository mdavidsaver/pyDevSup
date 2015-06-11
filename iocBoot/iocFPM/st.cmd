#!./bin/linux-x86/softIocPy

epicsEnvSet("ENGINEER","mdavidsaver")
epicsEnvSet("LOCATION","740")

# FPM waveform is 1M elements of type DBF_DOUBLE
epicsEnvSet("EPICS_CA_MAX_ARRAY_BYTES","9000000")

py "import logging"
py "logging.basicConfig(level=logging.DEBUG)"

#SR:C16-BI{FPM:1}BunchQ-Wf

dbLoadRecords("fpm.db","P=SR:C16-BI{FPM:2},INPA=SR:C16-BI{FPM:1}V-Wf CP MSI,INPD=SR:C03-BI{DCCT:1}I:Total-I MSI,NELM=1000000")
#dbLoadRecords("fpm.db","FNAME=fpm-5ma-teeth.npy,NELM=1000000")

dbLoadRecords("../../db/iocAdminSoft.db", "IOC=SR-CT{IOC:FPM}")
dbLoadRecords("../../db/save_restoreStatus.db", "P=SR-CT{IOC:FPM}")
save_restoreSet_status_prefix("SR-CT{IOC:FPM}")

#asSetFilename("/cf-update/acf/default.acf")
asSetFilename("/cf-update/acf/null.acf")

set_savefile_path("${PWD}/as","/save")
set_requestfile_path("${PWD}/as","/req")

set_pass0_restoreFile("ioc_settings.sav")

iocInit()

makeAutosaveFileFromDbInfo("as/req/ioc_settings.req", "autosaveFields_pass0")
create_monitor_set("ioc_settings.req", 10, "")

#caPutLogInit("ioclog.cs.nsls2.local:7004", 1)

# Start Reference tracker
#py "from devsup import disect; disect.periodic(10)"

dbl > records.dbl
#system "cp records.dbl /cf-update/$HOSTNAME.$IOCNAME.dbl"
