
class MySup(object):
    def __init__(self, rec):
        print rec, rec.field('VAL').fieldinfo()
        print 'VAL', rec.VAL
    def process(self, rec, reason):
        rec.VAL = 1+rec.VAL
    def detach(self, rec):
        print 'test1 detach',rec.name()

def build(rec, args):
	print 'test1 build for',rec.name()
	return MySup(rec)
