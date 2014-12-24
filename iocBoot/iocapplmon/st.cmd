#!../../bin/linux-x86/softIocPy2.7

< envPaths

#epicsEnvSet("APPLNAME", "arcapp01.cs.nsls2.local:17665")
epicsEnvSet("APPLNAME", "capp02.cs.nsls2.local:17665")

py "import bplreport"
py "bplreport.add('metrics','http://$(APPLNAME)/mgmt/bpl/getInstanceMetrics',3600)"
py "bplreport.add('typechange','http://$(APPLNAME)/mgmt/bpl/getPVsByDroppedEventsTypeChange',3600)"
py "bplreport.add('neverconn','http://$(APPLNAME)/mgmt/bpl/getNeverConnectedPVs',3600)"
py "bplreport.add('storage','http://$(APPLNAME)/mgmt/bpl/getStorageMetricsForAppliance?appliance=appliance0',3600)"

dbLoadRecords("../../db/applmetrics.db", "P=TST-CT{Arch:1}")
dbLoadRecords("../../db/bplstorage.db",  "P=TST-CT{Arch:1},N=0")
dbLoadRecords("../../db/bplstorage.db",  "P=TST-CT{Arch:1},N=1")
dbLoadRecords("../../db/bplstorage.db",  "P=TST-CT{Arch:1},N=2")

iocInit()
