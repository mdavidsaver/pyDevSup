
from zope.interface import Interface

class DeviceSupport(Interface):
    def detach(record):
        """Disconnect from the record.

        This is the last method called.
        """

    def allowScan(record):
        """Return True to allow SCAN='I/O Intr'
        or False to prevent this.
        """

    def process(record, reason):
        """Callback for record processing action.
        """
