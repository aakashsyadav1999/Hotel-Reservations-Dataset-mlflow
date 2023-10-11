import logging
import os
from datetime import datetime

#Log File define moth, day, year, hour, month, second
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#Define Directory where we want to apply logging
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

#Joining Log File
LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

#Basic Config
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)