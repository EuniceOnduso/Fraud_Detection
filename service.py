import pandas as pd
import json
from flask import  jsonify

def changeJsonToDataFrame(data: list):
    data_frame =pd.DataFrame(data)
    data_frame = data_frame.drop(columns='Class', axis=1)
    return data_frame

def iscleanDataFrame(data_frame: pd.DataFrame):
    comparison = pd.read_csv('../archive/Book1.csv')
    nulls = data_frame.isnull().sum().sum()
    if(nulls > 0 and set(comparison.columns) != set(data_frame.columns)):
        return False
    else:
        return True

def myReturn(data_frame: pd.DataFrame, model):
    if(iscleanDataFrame(data_frame)):
        result = model.predict(data_frame)
        result = result[0]
        if result == 0:
            return jsonify({"response":"non-fraudulent"}), 200
        else:
            return jsonify({"response":"fraudulent"}), 200
        
    else:
        return "failed", 400