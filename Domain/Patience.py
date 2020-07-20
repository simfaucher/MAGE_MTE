"""utils: various utility functions used by imagenode and imagehub
Copyright (c) 2017 by Jeff Bass.
License: MIT, see LICENSE for more details.
"""

import sys
import time
import signal

class Patience:
    """Timing class using system ALARM signal.
    When instantiated, starts a timer using the system SIGALRM signal. To be
    used in a with clause to allow a blocking task to be interrupted if it
    does not return in specified number of seconds.
    See main event loop in Imagenode.py for Usage Example
    Parameters:
        seconds (int): number of seconds to wait before raising exception
    """
    class Timeout(Exception):
        pass

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm

    def raise_timeout(self, *args):
        raise Patience.Timeout()
