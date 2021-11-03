# -*- coding: utf-8 -*- 
"""
@Author : rysong
@software: pycharm
@File : evaluate
@Time : 2021/9/24 4:51 下午 
@DESCRIPTION ：
沉着冷静，逻辑清晰
"""
import torch
import logging
from utils.function_utils import find_tag,Logger,logger,calc_metric
"""
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("test-1.log")  
logger.addHandler(file_handler)  
"""



def count(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]。tuple=(i_th of one batch, t_start, t_end, t_type_str)
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

   # y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true:
            num_correct += 1
    return num_proposed, num_correct, num_gold





def log_print(task_type,num_proposed_d, num_proposed, num_gold_d, num_gold, num_correct_d, num_correct, p_d, r_d, f1_d, p, r, f1):
    #logger = Logger('{}.log'.format(task_type))
    logger.info("detection")
    logger.info("proposed: {}\tcorrect: {}\tgold: {}".format(num_proposed_d, num_correct_d, num_gold_d))
    logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(p_d, r_d, f1_d))
    logger.info("identification")
    logger.info("proposed: {}\tcorrect: {}\tgold: {}".format(num_proposed, num_correct, num_gold))
    logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(p, r, f1))
    #logger.remove()

def evaluate(model, test_iterator, config, task_type):
    model.eval()
    with torch.no_grad():
        all_num_proposed, all_num_gold, all_num_correct, = 0, 0, 0
        all_num_proposed_d, all_num_gold_d, all_num_correct_d, = 0, 0, 0
        for i, batch in enumerate(test_iterator):
            token, seq_len, input_ids, attention_mask = batch['all_token'], batch['all_seq_len'], batch[
                'all_input_ids'].to(config.device), batch['all_attention_mask'].to(
                config.device)
            if task_type=='trigger':
                labels=batch['all_trigger_labels'].to(config.device)
                labels_hat,loss=model(input_ids, attention_mask, labels)
            else:
                trigger_index = batch["all_trigger_index"].to(config.device)
                labels=batch['all_argument_labels'].to(config.device)
                labels_hat, loss = model(input_ids, attention_mask, trigger_index, labels)

            trigger, argument, trigger_d, argument_d = find_tag(labels,task_type)
            trigger_hat, argument_hat, trigger_hat_d, argument_hat_d = find_tag(labels_hat,task_type)
            if task_type == 'trigger':
                num_proposed, num_correct, num_gold = count(trigger, trigger_hat)
                num_proposed_d, num_correct_d, num_gold_d = count(trigger_d, trigger_hat_d)
            else:
                num_proposed, num_correct, num_gold = count(argument, argument_hat)
                num_proposed_d, num_correct_d, num_gold_d = count(argument_d, argument_hat_d)
            all_num_proposed += num_proposed
            all_num_correct += num_correct
            all_num_gold += num_gold
            all_num_proposed_d += num_proposed_d
            all_num_gold_d += num_gold_d
            all_num_correct_d += num_correct_d
    p_d, r_d, f1_d = calc_metric(all_num_proposed_d, all_num_gold_d, all_num_correct_d)
    p, r, f1 = calc_metric(all_num_proposed, all_num_gold, all_num_correct)
    log_print(task_type,all_num_proposed_d, all_num_proposed, all_num_gold_d, all_num_gold, all_num_correct_d, all_num_correct,
              p_d, r_d, f1_d, p, r, f1)
    #获取logger的方法在子线程中，
    return p,r,f1
