# -*- coding: utf-8 -*- 
"""
@Author : rysong
@software: pycharm
@File : function_utils
@Time : 2021/9/22 10:13 上午 
沉着冷静，逻辑清晰
"""
import torch
import numpy as np
import time
from datetime import timedelta
from datasets.dataset import id2trigger_label, id2argument_label
import logging
import random

def seed_everything(seed):
    '''
        设置整个开发环境的seed
        :param seed:
        :param device:
        :return:
        '''
    #os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

logger=logging.getLogger()
class Logger(object):
    def __init__(self,file_name):
        self.logger = logging.getLogger()
        self.file_handler = logging.FileHandler(file_name)
        self.formatter = logging.Formatter('%(name)-12s %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                                      datefmt='%Y/%m/%d,%H:%M:%S')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.setLevel('INFO')
    def info(self,content):
        self.logger.info(content)
    def remove(self):
        self.logger.removeHandler(self.file_handler)

"""
def init_logger(file_name):
    logger=logging.getLogger()
    file_handler=logging.FileHandler(file_name)
    formatter=logging.Formatter('%(name)-12s %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel('INFO')
    logger.removeHandler(stream_handler)
    return  logger
"""

def find_tag(labels, mode):
    """
    :param : tensor.shape(bs,pad_size)
    :return: [(j_th,0, 2, 操作)]
             [(j_th,3, 4, Arg-obj),(j_th,6,9,Arg-stat]
    """
    if mode == 'trigger':
        id2label = id2trigger_label
    else:
        id2label = id2argument_label
    labels=labels.tolist()
    triggers,arguments,triggers_d,arguments_d=[],[],[],[]
    for j in range(len(labels)):
        labels[j] = [id2label[label].split('-', 2) for label in labels[j]]
        #labels[j]=嵌套列表

        result_trigger = []
        result_argument = []

        for i in range(len(labels[j])):  # len(labels)=128
            if labels[j][i][0] == 'B':
                if labels[j][i][1] == 'T':
                    result_trigger.append([i, i + 1, labels[j][i][2]])
                elif labels[j][i][1] == 'A':
                    result_argument.append([i, i + 1, labels[j][i][2]])

        for item in result_trigger:
            i = item[1]
            while i < len(labels[j]):
                if labels[j][i][0] == 'I' and labels[j][i][1] == 'T':
                    i = i + 1
                    item[1] = i
                else:
                    break

        for item in result_argument:
            i = item[1]
            while i < len(labels[j]):
                if labels[j][i][0] == 'I' and labels[j][i][1] == 'A':
                    i = i + 1
                    item[1] = i
                else:
                    break

        trigger=[(j,item[0], item[1],item[2]) for item in result_trigger]
        argument=[(j,item[0], item[1],item[2]) for item in result_argument]
        trigger_d = [(j,item[0], item[1]) for item in result_trigger]
        argument_d = [(j,item[0], item[1]) for item in result_argument]
        triggers.extend(trigger)
        arguments.extend(argument)
        triggers_d.extend(trigger_d)
        arguments_d.extend(argument_d)
    return triggers,arguments,triggers_d,arguments_d

def calc_metric(num_proposed, num_gold, num_correct):
    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 0.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 0.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1