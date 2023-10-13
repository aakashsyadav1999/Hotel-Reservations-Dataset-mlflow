from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Hotel_Reservation_Predicition.pipelines.prediction_pipeline import CustomData,PredictPipeline
import template

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            no_of_adults=int(request.form.get('no_of_adults')),
            no_of_children=int(request.form.get('no_of_children')),
            no_of_weekend_nights=int(request.form.get('no_of_weekend_nights')),
            no_of_week_nights=int(request.form.get('no_of_week_nights')),
            type_of_meal_plan=request.form.get('type_of_meal_plan'),
            required_car_parking_space=int(request.form.get('required_car_parking_space')),
            room_type_reserved=request.form.get('room_type_reserved'),
            lead_time=int(request.form.get('lead_time')),
            arrival_year=int(request.form.get('arrival_year')),
            arrival_month=int(request.form.get('arrival_month')),
            arrival_date=int(request.form.get('arrival_date')),
            market_segment_type=request.form.get('market_segment_type'),
            repeated_guest=int(request.form.get('repeated_guest')),
            no_of_previous_cancellations=int(request.form.get('no_of_previous_cancellations')),
            no_of_previous_bookings_not_canceled=int(request.form.get('no_of_previous_bookings_not_canceled')),
            avg_price_per_room=float(request.form.get('avg_price_per_room')),
            no_of_special_requests=int(request.form.get('no_of_special_requests')),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0")
    app.run()        

# from src.Hotel_Reservation_Predicition.logger import logging
# from src.Hotel_Reservation_Predicition.exception import CustomException
# from src.Hotel_Reservation_Predicition.components.data_ingestion import DataIngestion,DataIngestionConfig
# from src.Hotel_Reservation_Predicition.components.data_transformation import DataTransformation,DataTransformationConfig
# from src.Hotel_Reservation_Predicition.components.model_tranier import ModelTrainer,ModelTrainerConfig

# import sys


# if __name__=="__main__":
#     logging.info("The execution has started")

#     try:
#         #data_ingestion_config=DataIngestionConfig()
#         data_ingestion=DataIngestion()
#         train_data_path,test_data_path=data_ingestion.initiate_data_config()

#         #data_transformation_config=DataTransformationConfig()
#         data_transformation=DataTransformation()
#         train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

#         #model trainer
#         model_trainer=ModelTrainer()
#         print(model_trainer.initiate_model_trainer(train_arr,test_arr))

#     except Exception as e:
#         raise CustomException(e,sys)