import logging
import os
from datetime import datetime

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = f"log_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



'''
if __name__ == "__main__":
    logging.info("Logging has started.")

If you run python src/logger.py, then __name__ is "__main__" and this line runs:
logging.info("Logging has started.")

If another file imports logger (for example from src.logger import logging), t
hen __name__ is not "__main__", so that test log line does not run.


'''