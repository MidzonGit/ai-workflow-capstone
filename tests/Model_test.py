# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 11:09:09 2021

@author: MithunMohan
"""
import os
import unittest
from utils.model import model_train,model_load,model_predict

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def model_train_test(self):
        model_train(os.path.join("data","cs-train"),test=True)
        saved_model = os.path.join("models","sl-all-0_1.joblib")
        self.assertTrue(os.path.exists(saved_model))

    def model_load_test(self):
        all_data, all_models = model_load()
        model = all_models['all']
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    def model_predict_test(self):
        result = model_predict('all','2018','09','04',test=True) #change data
        y_pred = result['y_pred']
        self.assertTrue(y_pred >= 0.0)


if __name__ == '__main__':
    unittest.main()