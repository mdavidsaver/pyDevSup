
from zope.interface import Interface

class DeviceSupport(Interface):
    def detach(record):
        """Disconnect from the record.

        This is the last method called.
        """

    def allowScan(record):
        """Return True to allow SCAN='I/O Intr'
        or False to prevent this.
        
        If a callable object is returned then if
        will be invoked when I/O Intr scanning
        is disabled.  A Record instance is passed
        as the first (and only) argument.
        """

    def process(record, reason):
        """Callback for record processing action.
        """
