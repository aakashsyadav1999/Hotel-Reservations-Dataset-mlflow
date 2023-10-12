import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass


from src.Hotel_Reservation_Predicition.exception import CustomException
from src.Hotel_Reservation_Predicition.logger import logging


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import TargetEncoder


import os
from src.Hotel_Reservation_Predicition.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obg_file=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):

        try:
            cat_cols=[
                'type_of_meal_plan',
                'room_type_reserved',
                'market_segment_type'
            ]

            num_cols=[
                'no_of_adults',
                'no_of_children',
                'no_of_weekend_nights',
                'no_of_week_nights',
                'required_car_parking_space',
                'lead_time',
                'arrival_year',
                'arrival_month',
                'arrival_date',
                'repeated_guest',
                'no_of_previous_cancellations',
                'no_of_previous_bookings_not_canceled',
                'avg_price_per_room',
                'no_of_special_requests'
            ]
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("One_hot_encoder",OneHotEncoder()),
            ("StandardScalar", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{cat_cols}")
            logging.info(f"Numerical Columns:{num_cols}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_cols),
                    ("cat_pipeline",cat_pipeline,cat_cols)
                ]

            )
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='booking_status'


            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_tranformation_config.preprocessor_obg_file,
                obj=preprocessing_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obg_file
            )


        except Exception as e :
            raise CustomException(e,sys)
        

        