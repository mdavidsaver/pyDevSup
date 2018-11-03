#!./bin/linux-x86/softIocPy

py "import logging"
py "logging.basicConfig(level=logging.DEBUG)"

py "import sys; sys.path.insert(0,'${PWD}/testApp')"
py "print sys.path"

#py "import devsup.hooks"
#py "devsup.hooks.debugHooks()"

py "import test2"
py "test2.addDrv('AAAA')"
py "test2.addDrv('BBBB')"

py "import test6"
py "test6.SumTable(name='tsum')"

dbLoadRecords("db/test.db","P=md:")
dbLoadRecords("db/test6.db","P=tst:,TNAME=tsum")

iocInit()

# Start Reference tracker
#py "from devsup import disect; disect.periodic(10)"
