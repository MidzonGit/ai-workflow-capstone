# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:47:16 2021
modify this script
@author: MithunMohan
"""

import time,os
#from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")
    
import logging
formatter = logging.Formatter('%(asctime)s : %(filename)s : %(levelname)s : %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def update_train_log(msg):  
    log_msg=msg
    train_logger = setup_logger('train', 'logs/train.log')
    train_logger.info(str(log_msg))
    

def update_predict_log(msg):    
    log_msg=msg
    predict_logger = setup_logger('predict', 'logs/predict.log')
    predict_logger.info(str(log_msg))
    

if __name__ == "__main__":
    update_train_log('Train logger')
    update_predict_log("Predict logger")