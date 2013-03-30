
class MySup(object):
    def __init__(self, rec):
        self.val = rec.field('VAL')
    def process(self, rec, reason):
        print 'test1 proc',rec.name(),reason
        self.val.putval(1+self.val.getval())
    def detach(self, rec):
        print 'test1 detach',rec.name()

def build(rec, args):
	print 'test1 build for',rec.name()
	return MySup(rec)
