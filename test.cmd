#!./bin/linux-x86-debug/devsup
dbLoadDatabase("dbd/devsup.dbd")
devsup_registerRecordDeviceDriver(pdbbase)

evalPy "print 1"
evalPy "print 2"
