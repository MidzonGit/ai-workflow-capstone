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


def update_train_log(tag,period,rmse,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=False):  
    log_msg=''
    train_logger = setup_logger('train', 'logs/train.log')
    train_logger.info(str(log_msg))
    

def update_predict_log(country, y_pred,y_proba,target_date,runtime,MODEL_VERSION,test=False):    
    log_msg=''
    predict_logger = setup_logger('predict', 'logs/predict.log')
    predict_logger.info(str(log_msg))
    

if __name__ == "__main__":
    from model import MODEL_VERSION, MODEL_VERSION_NOTE
    update_train_log(str((100,10)),"{'rmse':0.5}","00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
    update_predict_log("[0]","[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
                       "00:00:01",MODEL_VERSION, test=True)