import utils
from utils import *

import train_test
from train_test import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Baheya_data = Load_data('Datasets/Merged_baheya.csv')

X_val, y_val = prepare_Rec_data(Baheya_data) # Baheya

columns_to_encode = ['Menopausal Status','Tumor Size','Lymph Node Status', 'Tumor Grade', 'Histological Type',
                     'ER', 'PR','HER2','Tumor Location','Overall Survival Status']

X_val = one_hot_encode_columns(X_val, columns_to_encode) # Baheya

y_val = label_encode_target(y_val) # Baheya

X_val_scaled = train_test.scaler.transform(X_val[['Overall Survival Status (Month)', 'Recurrence Free Status (Month)']])

loaded_model_xgb = joblib.load('Saved Models/model_Rec_xgb.pkl')
loaded_model_rf = joblib.load('Saved Models/model_Rec_rf.pkl')
loaded_model_SV = joblib.load('Saved Models/model_Rec_SV.pkl')
loaded_model_NN = joblib.load('Saved Models/model_Rec_NN.pkl')

def validate_Rec_xgboost_model(X_val, y_val):
    # Training and Testing accuracy
    utils.evaluate_validation_accuracy(loaded_model_xgb, X_val, y_val, 'model_Rec_xgb_Val')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(loaded_model_xgb, X_val, y_val, 'model_Rec_xgb_Val')

def validate_Rec_rf_model(X_val, y_val):
    # Training and Testing accuracy
    utils.evaluate_validation_accuracy(loaded_model_rf, X_val, y_val, 'model_Rec_rf_Val')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(loaded_model_rf, X_val, y_val, 'model_Rec_rf_Val')

def validate_Rec_SV_model(X_val, y_val):
    # Training and Testing accuracy
    utils.evaluate_validation_accuracy(loaded_model_SV, X_val, y_val, 'model_Rec_SV_Val')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(loaded_model_SV, X_val, y_val, 'model_Rec_SV_Val')

def validate_Rec_NN_model(X_val, y_val):
    # Training and Testing accuracy
    utils.evaluate_validation_accuracy(loaded_model_NN, X_val, y_val, 'model_Rec_NN_Val')
    
    # Plot evaluation metrics
    utils.plot_evaluation_metrics(loaded_model_NN, X_val, y_val, 'model_Rec_NN_Val')


