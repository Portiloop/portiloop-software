import logging
import sys
from pathlib import Path


__version__ = '0.1.4'


LOG_FOLDER = Path.home() / 'workspace' / 'logs'
LOG_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_FOLDER / 'portiloop.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(processName)s %(name)s (%(filename)s:%(lineno)d): %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
