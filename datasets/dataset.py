# -*- coding: utf-8 -*- 
"""
@Author : rysong
@software: pycharm
@File : dataset
@Time : 2021/11/2 9:18 下午 
@DESCRIPTION ：
沉着冷静，逻辑清晰
"""

import torch
import json
from data import const
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset, TensorDataset


def label2id(event_types, argument_roles, BIO_tagging=True):
    trigger_labels = ['O']
    argument_labels = ['O']
    for event_type in event_types:
        if BIO_tagging:
            label_t = 'B-T-{}'.format(event_type)
            if label_t not in trigger_labels:
                trigger_labels.append(label_t)
                trigger_labels.append('I-T-{}'.format(event_type))
        else:
            trigger_labels.append(event_type)
    for argument_role in argument_roles:
        if BIO_tagging:
            label_e = 'B-A-{}'.format(argument_role)
            if label_e not in argument_labels:
                argument_labels.append(label_e)
                argument_labels.append('I-A-{}'.format(argument_role))
        else:
            argument_labels.append(argument_role)
    trigger_label2idx = {tag: idx for idx, tag in enumerate(trigger_labels)}
    argument_label2idx = {tag: idx for idx, tag in enumerate(argument_labels)}
    idx2trigger_label = {idx: tag for idx, tag in enumerate(trigger_labels)}
    idx2argument_label = {idx: tag for idx, tag in enumerate(argument_labels)}
    event_types2idx = {tag: idx for idx, tag in enumerate(event_types)}
    return trigger_label2idx, argument_label2idx, idx2trigger_label, idx2argument_label, event_types2idx


trigger_label2id, argument_label2id, id2trigger_label, id2argument_label, event_type2id = label2id(
    event_types=const.event_types,
    argument_roles=const.argument_roles)


def role2label(roles, label2id, pad_size):
    labels = []
    for i in roles:
        id = label2id[i]
        labels.append(id)
    # labels = [label2id[i] for i in roles]
    if pad_size:
        if len(labels) < pad_size:

            labels += ([0] * (pad_size - len(labels)))
        else:

            labels = labels[:pad_size]
    return labels


def load_data(path, config):
    config.token_nizer = BertTokenizer.from_pretrained(config.bert_path)
    examples = []
    cut_off = config.pad_size
    with open(path, 'r',encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            words = [item['sentence'][i] for i in range(len(item['sentence']))]  # sentence的每个字
            token = []
            for w in words:
                if w==' ':
                    w=","
                t = config.token_nizer.tokenize(w)  # 返回list
                token.extend(t)  # append报错 list not hashable
            token = ['[CLS]'] + token + ['[SEP]']
            attention_mask = []
            input_ids = config.token_nizer.convert_tokens_to_ids(token)
            if config.pad_size:
                if len(token) < config.pad_size:  # 填充
                    attention_mask = [1] * len(input_ids) + [0] * (config.pad_size - len(token))
                    input_ids += ([0] * (config.pad_size - len(token)))
                    seq_len = len(token)
                else:  # 截断
                    attention_mask = [1] * config.pad_size
                    input_ids = input_ids[:config.pad_size]
                    seq_len = config.pad_size
            example = {'token':token,'input_ids': input_ids, 'attention_mask': attention_mask, 'seq_len': seq_len}
            trigger_roles = ['O' for _ in range(len(token))][:cut_off]
            trigger_argument_items = {}
            try:
                for event_item in item['events']:
                    for trigger_mention in event_item['trigger']:#trigger_mention only one
                        start2 = trigger_mention['start']
                        end2 = trigger_mention["end"]
                        start2 = int(start2)
                        end2 = int(end2)
                        if start2 >= cut_off:
                            continue
                        for i in range(start2, min(end2, cut_off - 1)):
                            event_type = trigger_mention['event_type']
                            if i == start2:

                                trigger_roles[i + 1] = 'B-T-{}'.format(event_type)

                            else:

                                trigger_roles[i + 1] = 'I-T-{}'.format(event_type)
                    #trigger_index = event_type2id[event_type]
                    trigger_index=(min(start2+1,cut_off-1),min(end2,cut_off-1))
                    argument_roles = ['O' for _ in range(len(token))][:cut_off]
                    for argument_mention in event_item['arguments']:  # entity_mention字典
                        start1 = argument_mention['start']
                        end1 = argument_mention["end"]
                        start1 = int(start1)
                        end1 = int(end1)
                        if start1 >= cut_off:
                            continue
                        end1 = min(end1, cut_off - 1)
                        for i in range(start1, end1):
                            role = argument_mention['role']
                            if i == start1:
                                role = 'B-A-{}'.format(role)
                            else:
                                role = 'I-A-{}'.format(role)
                            argument_roles[i + 1] = role

                    argument_labels = role2label(argument_roles, argument_label2id, config.pad_size)
                    trigger_argument_items[trigger_index] = argument_labels
            except:
                continue

            trigger_labels = role2label(trigger_roles, trigger_label2id, config.pad_size)
            example['trigger_labels'] = trigger_labels
            example['trigger_argument_items'] = trigger_argument_items
            examples.append(example)
        return examples



class TriggerDataset(TensorDataset):
    def __init__(self, path, config):
        examples = load_data(path, config)
        trigger_examples = []
        for example in examples:
            trigger_example = {}
            trigger_example['token']=example['token']
            trigger_example['seq_len']=example['seq_len']
            trigger_example['input_ids'] = example['input_ids']
            trigger_example['attention_mask'] = example['attention_mask']
            trigger_example['trigger_labels'] = example['trigger_labels']
            trigger_examples.append(trigger_example)
        self.trigger_examples = trigger_examples
        self.len = len(examples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.trigger_examples[index]


class ArgumentDataset(Dataset):
    def __init__(self, path, config):
        examples = load_data(path, config)
        argument_examples=[]
        for example in examples:
            trigger_argument_items = example['trigger_argument_items']  # dict
            for k, v in trigger_argument_items.items():
                argument_example = {}
                argument_example['token']=example['token']
                argument_example['seq_len']=example['seq_len']
                argument_example['input_ids']=example['input_ids']
                argument_example['attention_mask']=example['attention_mask']
                argument_example['trigger_index']=k #tuple
                argument_example['argument_labels']=v #list
                argument_examples.append(argument_example)
        self.argument_examples=argument_examples

    def __len__(self):
        return len(self.argument_examples)

    def __getitem__(self, index):
        return self.argument_examples[index]


def collate_fn_t(batch):
    """

    :param batch: dataloder读取的一batch的dataset
    :return:
    """
    all_token=[x['token'] for x in batch]
    all_seq_len=[x['seq_len'] for x in batch]
    all_input_ids = torch.tensor([x['input_ids'] for x in batch]) #嵌套列表
    all_attention_mask = torch.tensor([x['attention_mask'] for x in batch])
    all_labels = torch.tensor([x["trigger_labels"] for x in batch])

    return {
        "all_token":all_token,
        "all_seq_len":all_seq_len,
        "all_input_ids": all_input_ids,
        "all_attention_mask": all_attention_mask,
        "all_trigger_labels": all_labels,
    }

def collate_fn_a(batch):
    all_token = [x['token'] for x in batch]
    all_seq_len = [x['seq_len'] for x in batch]
    all_input_ids = torch.tensor([x['input_ids'] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'] for x in batch])
    all_trigger_index = torch.tensor([x["trigger_index"] for x in batch])
    all_argument_labels=torch.tensor([x['argument_labels'] for x in batch])
    return {
        "all_token": all_token,
        "all_seq_len": all_seq_len,
        "all_input_ids": all_input_ids,
        "all_attention_mask": all_attention_mask,
        "all_trigger_index": all_trigger_index,
        "all_argument_labels":all_argument_labels
    }

