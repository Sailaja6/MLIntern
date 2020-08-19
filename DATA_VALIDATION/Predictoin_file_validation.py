import os
import pandas as pd
import numpy as np
import json
import shutil



class Prediction_validation:
    """"
    This class is used for validating the raw training data.
    parameters: File Path containing Raw training data
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def getvaluesfromjson(self):
        """"
        Description: To get the details of data from .json file
        Input: None
        Output: no. of columns, column names
        """
        json_file = open("schema_training.json", 'r')
        d = json.load(json_file)
        json_file.close()
        col_no = d['NumberofColumns']
        col_names = d['colName'].keys()
        col_names = list(col_names)
        col_info = d['colName']
        return col_no, col_names, col_info

    def create_prediction_directory(self):
        """"
        Description: Delete Prediction batches directory, if exist, else  create new directories
               Prediction_batches/Good_data ->>For validated files
               Prediction_batches/Bad_data ->>For non validated files

        Input: None
        Output: None
        """
        if os.path.isdir('Prediction_batches'):  #Check existing directory
            shutil.rmtree('Prediction_batches')
            os.makedirs('Prediction_batches/Good_data')  # Creating new directory
            os.makedirs('Prediction_batches/Bad_data')

        else:
            os.makedirs('Prediction_batches/Good_data') # Creating new directory
            os.makedirs('Prediction_batches/Bad_data')

    def col_length_validation(self, col_no):
        """"
        Description: Check the column length of raw data. If equal to "col_no" move the file to
              'Training_batches/Good_data' , else, move to 'Training_batches/Bad_data'

        Input: col_no
        Output: None
        """
        for file in os.listdir(self.file_path):
            if file.split('.')[1] == 'csv':
                data = pd.read_csv(os.path.join(self.file_path, file), header=None)
                if data.shape[1] == col_no:
                    shutil.copy(os.path.join(self.file_path, file), 'Prediction_batches/Good_data')
                else:
                    shutil.copy(os.path.join(self.file_path, file), 'Prediction_batches/Bad_data')

    def missing_value_validation(self):
        """"
        Description: Used to check missing values in the column. If there is no missing values move the file to
        Good_data folder , else to bad data folder
        """
        for file in os.listdir('Prediction_batches/Good_data'):
            data = pd.read_csv(os.path.join('Prediction_batches/Good_data', file), header=None)
            if data.isna().sum().sum() != 0:
                shutil.move(os.path.join('Prediction_batches/Good_data', file)   #move null value files from Good_Data to
                            , os.path.join('Prediction_batches/Bad_data', file))  # Bad_data folder
                temp = data.isna().sum()
                message = "Null values in {} at cols {}".format(file, list(temp[temp != 0].index))


