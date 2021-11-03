# -*- coding: utf-8 -*- 
"""
@Author : rysong
@software: pycharm
@File : test
@Time : 2021/9/23 11:30 上午 
沉着冷静，逻辑清晰
1 以事件为单位（属性全部正确算一个事件正确），以f1为指标，测试
2 单独测试各属性
"""
import json
import os
from utils.config import config
from utils.function_utils import Logger, seed_everything
from utils.function_utils import calc_metric

def load_json(path):
    """
    读取json文件，返回事件列表和各个属性列表（t a)
    :param path:
    :return: text_list, which index=评测给出的test数据的顺序索引
    trigger_list/argument_list,shape=(ith，start,end,event_type/argument_role)
    """
    text_list,trigger_list,argument_list,argument_t_list=[],[],[],[]
    with open(path,'r',encoding='utf-8') as f:
        items=json.load(f)
        text_list=[]
        for i in range(len(items)):
            item=items[i]
            events_list=[]
            for event_item in item['events']:
                trigger_item=event_item["trigger"][0] #dict
                trigger_dict ={"start":int(trigger_item["start"]),"end":int(trigger_item["end"]),"role":trigger_item["event_type"]}
                trigger_list.append((i,int(trigger_item["start"]),int(trigger_item["end"]),trigger_item["event_type"]))
                argument_=set()
                for argument_item in event_item["arguments"]: #dict
                    argument_l=(int(argument_item["start"]),int(argument_item["end"]),argument_item["role"])
                    argument_.add(argument_l)
                    argument_list.append((i,int(argument_item["start"]),int(argument_item["end"]),argument_item["role"]))
                    argument_t_list.append((i,int(argument_item["start"]),int(argument_item["end"]),argument_item["role"],trigger_item["event_type"]))
                events_list.append({"trigger":trigger_dict,"arguments":argument_})
            text_list.append(events_list)
    trigger_d_list = [(item[0], item[1], item[2]) for item in trigger_list]
    argument_d_list = [(item[0], item[1], item[2]) for item in argument_list]
    argument_t_d_list=[ (item[0], item[1], item[2],item[4]) for item in argument_t_list]

    return text_list,trigger_list,argument_list,trigger_d_list,argument_d_list,argument_t_list,argument_t_d_list



def calc_event(predict_text_list,gold_text_list):
    """
    以事件为单位（属性全部正确算一个事件正确,返回p r f1
    :param predict_text_list:
    :param gold_text_list:
    :return:p,r,f1
    list.item=[{"trigger":[],
                "arguments":[]
                },
                {}]
    """
    num_proposed,num_correct,num_gold=0,0,0
    for i in range(len(predict_text_list)):
        num_gold += len(gold_text_list[i])
        num_proposed += len(predict_text_list[i])
        for predict_event_item in predict_text_list[i]:
            if predict_event_item in gold_text_list[i]:
                num_correct+=1
    print("包含事件")
    print(num_proposed,num_gold,num_correct)
    return calc_metric(num_proposed,num_gold, num_correct)

def calc_trigger(predict_trigger_list,gold_trigger_list):
    """

    :param predict_trigger_list:
    :param gold_trigger_list:
    :return: p,r,f1
    """
    num_proposed=len(predict_trigger_list)
    num_gold=len(gold_trigger_list)
    #print(gold_trigger_list)
    #print(predict_trigger_list)
    num_correct=0
    for item in predict_trigger_list:
        if item in gold_trigger_list:
            num_correct+=1
    return calc_metric(num_proposed,num_gold,num_correct)

def calc_argument(predict_argument_list,gold_argument_list,mode):
    """

    :param predict_argument_list:
    :param gold_argument_list:
    :param mode: =0: 返回所有a合在一起的p r f1;
                =1: 返回分别每种a的p r f1
    :return:
    """
    if mode==0:
        num_propose=len(predict_argument_list)
        num_gold=len(gold_argument_list)
        num_correct=0
        for item in predict_argument_list:
            if item in gold_argument_list:
                num_correct+=1
        return calc_metric(num_propose,num_gold,num_correct)
    if mode==1:
        num_propose_obj,num_propose_stat,num_propose_result,num_gold_obj,num_gold_stat,num_gold_result,\
        num_correct_obj,num_correct_stat,num_correct_result=0,0,0,0,0,0,0,0,0
        for item in predict_argument_list:
            #item=(i,start,end,role)
            if item[3]=='Arg-stat':
                num_propose_stat+=1
                if item in gold_argument_list:
                    num_correct_stat+=1
            elif item[3]=='Arg-obj':
                num_propose_obj+=1
                if item in gold_argument_list:
                    num_correct_obj+=1
            else:
                num_propose_result+=1
                if item in gold_argument_list:
                    num_correct_result+=1
        for item in gold_argument_list:
            if item[3]=='Arg-stat':
                num_gold_stat+=1
            elif item[3]=='Arg-obj':
                num_gold_obj+=1
            else:
                num_gold_result+=1
        p_obj,r_obj,f_obj=calc_metric(num_propose_obj,num_gold_obj,num_correct_obj)
        p_stat, r_stat, f_stat = calc_metric(num_propose_stat, num_gold_stat, num_correct_stat)
        p_result, r_result, f_result = calc_metric(num_propose_result, num_gold_result, num_correct_result)
        return p_obj,r_obj,f_obj,p_stat, r_stat, f_stat,p_result, r_result, f_result

if __name__ == '__main__':
    config=config()

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = Logger('logs/test.log')
    logger.info("\n\n*******************This is a new test log！")

    precdict_text_list, predict_trigger_list, predict_argument_list, precdict_trigger_d_list,predict_argument_d_list,predict_argument_t_list,predict_argument_t_d_list=load_json(config.predict_path)
    text_list, trigger_list, argument_list,trigger_d_list,argument_d_list,argument_t_list,argument_t_d_list = load_json(config.test_path)

    if config.do_ace_test:
        t_d_p, t_d_r, t_d_f1 = calc_trigger(precdict_trigger_d_list, trigger_d_list)
        t_p, t_r, t_f1 = calc_trigger(predict_trigger_list, trigger_list)
        a_d_p, a_d_r, a_d_f1 = calc_argument(predict_argument_d_list, argument_d_list, 0)
        a_p, a_r, a_f1 = calc_argument(predict_argument_list, argument_list, 0)
        logger.info("以ACE指标测试")
        logger.info("trigger_detection")
        logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(t_d_p, t_d_r, t_d_f1))
        logger.info("trigger_identification")
        logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(t_p, t_r, t_f1))
        logger.info("argument_detection")
        logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(a_d_p, a_d_r, a_d_f1))
        logger.info("argument_identification")
        logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(a_p, a_r, a_f1))


    if config.do_event_test:
        e_p,e_r,e_f1=calc_event(precdict_text_list,text_list)
        logger.info("以事件为单位测试")
        logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(e_p, e_r, e_f1))

    '''
    单test argument,带上event_type
    p,r,f1=calc_argument(predict_argument_t_list,argument_t_list,0)
    logger.info("测试多事件下的论元")
    logger.info("P={:.3f}\tR={:.3f}\tF1={:.3f}".format(p,r,f1))
    '''


