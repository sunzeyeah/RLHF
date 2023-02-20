
import os
import logging.config

from src.utils.config import RESOURCE_PATH

logging.config.fileConfig(os.path.join(RESOURCE_PATH, "config", "log.conf"), disable_existing_loggers=False)
logger = logging.getLogger(__name__)
