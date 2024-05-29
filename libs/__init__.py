import logging
import config

logger = logging.getLogger()
logger.setLevel(config.log_level)
handler = logging.StreamHandler()
handler.setLevel(config.log_level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)