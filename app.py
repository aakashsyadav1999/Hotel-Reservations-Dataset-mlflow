from src.Hotel_Reservation_Predicition.logger import logging
from src.Hotel_Reservation_Predicition.exception import CustomException
from src.Hotel_Reservation_Predicition.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.Hotel_Reservation_Predicition.components.data_transformation import DataTransformation,DataTransformationConfig
from src.Hotel_Reservation_Predicition.components.model_tranier import ModelTrainer,ModelTrainerConfig

import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_config()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        #model trainer

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        raise CustomException(e,sys)