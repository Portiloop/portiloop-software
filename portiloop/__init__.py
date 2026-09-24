import logging
import logging.handlers
import sys
from pathlib import Path


__version__ = '0.1.4'


class SizeCappedFileHandler(logging.handlers.RotatingFileHandler):
    """
    Single log file that never grows beyond maxBytes.
    When the limit is reached, the oldest half of the file is discarded instead of creating backup files.
    """
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        # Trim in place (same file) so that other processes appending to this file keep working
        with open(self.baseFilename, 'rb+') as f:
            size = f.seek(0, 2)
            start = max(0, size - self.maxBytes // 2)
            f.seek(start)
            data = f.read()
            if start > 0:
                data = data[data.find(b'\n') + 1:]  # drop the partial first line
            f.seek(0)
            f.write(data)
            f.truncate()
        self.stream = self._open()


LOG_FOLDER = Path.home() / 'workspace' / 'logs'
LOG_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_FOLDER / 'portiloop.log'
LOG_MAX_BYTES = 10 * 1024  # 10 * 1024 * 1024
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d): %(message)s',
    handlers=[
        SizeCappedFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES),
        logging.StreamHandler(sys.stdout),
    ],
)
