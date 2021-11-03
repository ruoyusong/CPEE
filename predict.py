# -*- coding: utf-8 -*- 
"""
@Author : rysong
@software: pycharm
@File : predict
@Time : 2021/9/23 11:30 上午 
沉着冷静，逻辑清晰
"""
import json
import os

import torch
import traceback
from models.model import TriggerModel, ArgumentModel
from utils.config import config
from datasets.dataset import TriggerDataset, role2label, argument_label2id, collate_fn_a,collate_fn_t
from torch.utils.data import DataLoader, Dataset
from utils.function_utils import seed_everything
from utils.function_utils import find_tag


def predict_trigger(path, config):
    """

    :param path:
    :param config:
    :return: examples：list,类似load_data格式，后进一步生成iter
    trigger_num_counts:list,每个样本预测出的trigger个数
    """
    model = TriggerModel(config).to(config.device)  #!!!踩雷
    model.load_state_dict(torch.load(config.model_save_path+'/best_trigger_model.pth'))

    seed_everything(11)
    dataset = TriggerDataset(path, config)
    dataiter = DataLoader(dataset, shuffle=False, batch_size=config.batch_size,collate_fn=collate_fn_t)
    predict_json = []
    examples = []
    trigger_num_counts = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataiter):
            token, seq_len, input_ids, attention_mask = batch['all_token'], batch['all_seq_len'], batch[
                'all_input_ids'].to(config.device), batch['all_attention_mask'].to(
                config.device)
            trigger_labels = batch["all_trigger_labels"].to(config.device)
            trigger_labels_hat, _ = model(input_ids, attention_mask, trigger_labels)
            trigger_hat, _, trigger_hat_d, _ = find_tag(trigger_labels_hat, 'trigger')
            #print("预测出{}".format(len(trigger_hat)))
            # 预测的json格式,token嵌套列表
            for token_sentence in token:  # token 嵌套列表
                text = "".join(token_sentence)
                text=text.strip('[CLS]')
                text=text.strip('[SEP]')
                dict_sentence = {"sentence": text, "events": []}
                predict_json.append(dict_sentence)

            # 生成以句子为单位的数据集
            trigger_num_count = [0 for num in range(len(input_ids))]
            for t in trigger_hat:  # t=(j_th,0, 2, 操作) batch下的所有
                j = t[0]
                trigger_num_count[j] += 1
                trigger_index = [t[1], t[2]-1]
                example = dataset[i * config.batch_size + j].copy()  # dict.keys=token,seq_len,input_ids,,,

                argument_roles = ['O' for _ in range(example['seq_len'])]
                argument_labels = role2label(argument_roles, argument_label2id, config.pad_size)
                example['trigger_index'] = trigger_index
                example['argument_labels'] = argument_labels
                examples.append(example)

                #为predict_json添加预测的trigger
                sentence=predict_json[i*config.batch_size+j]['sentence']
                trigger_item=[{"text":sentence[t[1]-1:t[2]-1],
                               "start":t[1]-1,
                               "end":t[2]-1,
                               "event_type":t[3]}]
                predict_json[i*config.batch_size+j]['events'].append({"trigger":trigger_item,"arguments":[]})
            trigger_num_counts.extend(trigger_num_count)

    return examples, trigger_num_counts,predict_json   #错误分析：没有预测出trigger的也应该包含在examples


class Predict_dataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

def sentence_index(trigger_num_counts,num):
    """
    返回当前argument属于第几句和当前句子的第几个事件（均0开始）
    :param trigger_num_counts:
    :param num: 在argument_iter中的index
    :return: 在trigger_iter中的index,即属于predict_json的index
    在event_list的index
    """
    nums=0
    for i in range(len(trigger_num_counts)):
        if trigger_num_counts[i]==0:
            continue
        else:
            nums_before=nums #[0，i-1]的所有trigger个数
            nums+=trigger_num_counts[i]#[0,i]的所有trigger个数
            if num>=nums_before and num<nums:
                break
    return i,num-nums_before


def predict_arguments(examples, trigger_num_counts, predict_json, config):
    dataset = Predict_dataset(examples)
    dataiter = DataLoader(dataset=dataset,
                          shuffle=False,
                          batch_size=config.batch_size,
                          collate_fn=collate_fn_a)

    model = ArgumentModel(config).to(config.device)
    model.load_state_dict(torch.load(config.model_save_path+'/best_argument_model.pth'))
    seed_everything(11)
    model.eval()
    with torch.no_grad():
        predict_result = []
        for i, batch in enumerate(dataiter):
            token, seq_len, input_ids, attention_mask = batch['all_token'], batch['all_seq_len'], batch[
                'all_input_ids'].to(config.device), batch['all_attention_mask'].to(
                config.device)
            trigger_index = batch["all_trigger_index"].to(config.device) #tuple
            #print(trigger_index)
            #print('\n')
            argument_labels = batch["all_argument_labels"].to(config.device)
            argument_hat, loss = model(input_ids, attention_mask, trigger_index, argument_labels)
            trigger, argument, trigger_d, argument_d = find_tag(argument_hat, 'argument')
            #print(argument)
            #print('\n')
            for a in argument: # a=(j_th,0, 2, 操作)
                try:
                    j=a[0]
                    p,q=sentence_index(trigger_num_counts,i*config.batch_size+j)
                    sentence = predict_json[p]['sentence']#属于当前batch的j_th句,属于测试集的第k句
                    argument_item={
                            "text": sentence[a[1]-1:a[2]-1],
                            "start": a[1]-1,
                            "end": a[2]-1,
                            "role": a[3]
                    }
                    predict_json[p]["events"][q]['arguments'].append(argument_item)
                except Exception as e:
                    print(a)
                    print("p:{}".format(p))
                    print("q:{}".format(q))
                    print(argument_item)
                    traceback.print_exc()
                    continue
    return predict_json

if __name__ == '__main__':
    config = config()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device=device
    examples, trigger_num_counts ,predict_json= predict_trigger(config.test_path,config)

    predict_json=predict_arguments(examples,trigger_num_counts,predict_json,config)
    f=open(config.predict_path, 'w+', encoding='utf-8')
    json.dump(predict_json,f,ensure_ascii=False,indent=2)