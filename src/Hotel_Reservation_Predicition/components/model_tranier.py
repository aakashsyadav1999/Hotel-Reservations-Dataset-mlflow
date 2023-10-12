import os 
import sys
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss

from src.Hotel_Reservation_Predicition.exception import CustomException
from src.Hotel_Reservation_Predicition.logger import logging
from src.Hotel_Reservation_Predicition.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info("Split training data and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
            }
            params={
                "DecisionTreeClassifier": {
                    'criterion':['gini'],
                    'max_features':[2,3,4,5,6,7],
                #    'max_depth':[10,11,12,13],
                },
                "RandomForestClassifier":{
                   'n_estimators' : [100],
                #    'n_jobs' : [-1],
                   'max_features':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                #   'max_depth':[3, 4, 5, 6, 7, 9, 11],
                #    'min_samples_split':[2,3]
                }
            }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            #To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get best name form dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            print("This is the best model")
            best_model=models[best_model_name]

            if best_model_score<0.2:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset {best_model}")
            logging.info(f"Best found model on both training and testing dataset {best_model.get_params()}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )

            predicted=best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            prec = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            return acc
            return prec
            return recall
            print(acc)
            print(prec)
            print(recall)
            logging.log(acc,prec,recall)


        except Exception as e:
            raise CustomException (e,sys)