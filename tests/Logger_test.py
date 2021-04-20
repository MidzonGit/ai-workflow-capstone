# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 11:11:55 2021

@author: MithunMohan
"""

import os
import csv
import unittest
from ast import literal_eval
import pandas as pd

## import model specific functions and variables
from logger import update_train_log, update_predict_log

class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_train(self):
        """
        train log testing
        """

        log_file = os.path.join("logs","train.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        ## update the log
        tag = 'united_kingdom'
        date = '2018-07-01'
        rmse = 0.9
        runtime = '00:05:26'
        model_version = 0.1
        model_version_note = "test model"
        msg = "{0} version {1} trained for {2} during {3} with accuracy {4}. Total runtime {5}".format(model_version_note,model_version,tag,date,rmse,runtime)
        update_train_log(msg)
        self.assertTrue(os.path.exists(log_file))
                

    def test_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs","predict.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        tag = 'eire'
        y_pred = [4]
        y_proba = [0.25,0.75]
        target_date = '2018-07-01'
        runtime = "00:00:01"
        msg = "Prediction generated for {0} with value {1}, probability {2} for {3}. Total runtime was {4}".format(tag,y_pred,y_proba,target_date,runtime)
        update_predict_log(msg)
        self.assertTrue(os.path.exists(log_file))


### Run the tests
if __name__ == '__main__':
    unittest.main()