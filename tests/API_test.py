# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 11:09:06 2021

@author: MithunMohan
"""

import sys
import os
import unittest
import requests
import json
import re
from ast import literal_eval
import numpy as np

port = 8080

try:
    requests.post('http://127.0.0.1:{}/'.format(port))
    server_available = True
except:
    server_available = False
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    testing the apis
    """

    @unittest.skipUnless(server_available,"local server is not running")
    def train_api_test(self):
        r = requests.post('http://127.0.0.1:{}/train'.format(port),json={"mode":"test"})
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_02_predict_empty(self):
        r = requests.post('http://127.0.0.1:{}/predict'.format(port))
        self.assertEqual(re.sub('\n|"','',r.text),"[]")

        ## provide improperly formatted data
        r = requests.post('http://127.0.0.1:{}/predict'.format(port),json={"key":"value"})     
        self.assertEqual(re.sub('\n|"','',r.text),"[]")
    
    @unittest.skipUnless(server_available,"local server is not running")
    def predict_api_test(self):
        request_json = {'country':'all','year':'2018','month':'05','day':'01','mode':'test'}
        r = requests.post('http://127.0.0.1:{}/predict'.format(port),json=request_json)
        
        response = json.loads(r.text)
        self.assertTrue(len(response['y_pred']) > 0)

        
### Run the tests
if __name__ == '__main__':
    unittest.main()