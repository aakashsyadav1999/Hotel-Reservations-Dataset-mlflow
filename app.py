from src.Hotel_Reservation_Predicition.logger import logging
from src.Hotel_Reservation_Predicition.exception import CustomException
from src.Hotel_Reservation_Predicition.components.data_ingestion import DataIngestion
from src.Hotel_Reservation_Predicition.components.data_ingestion import DataIngestionConfig

import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_config()

    except Exception as e:
        raise CustomException(e,sys)