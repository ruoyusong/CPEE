# -*- coding: utf-8 -*- 
"""
@Author : rysong
@software: pycharm
@File : config
@Time : 2021/9/19 6:36 下午 
沉着冷静，逻辑清晰
"""

import argparse
def config():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task_type',type=str,default='trigger',choices=['trigger','argument'],help="请输入训练的模块，trigger or argument")
    parser.add_argument("--batch_size",type=int,default=64,help="64")
    parser.add_argument("--learning_rate", type=float, default=3e-5,help="请输入学习率，经调参,trigger：5e-5,argument:3e-5")


    parser.add_argument("--train_path",type=str,default='data/train.json')
    parser.add_argument("--dev_path", type=str, default='data/dev.json')
    parser.add_argument("--test_path",type=str,default='data/test.json')
    parser.add_argument("--multi_event_path", type=str, default='data/multi_event.json')
    parser.add_argument("--model_save_path", type=str, default='result')
    parser.add_argument("--predict_path", type=str, default='data/predict.json')

    parser.add_argument("--num_epochs",type=int,default=30)
    parser.add_argument("--pad_size",type=int,default=128)
    parser.add_argument("--bert_path", type=str, default='/home/rysong/work/pretrained_model/bert_base_chinese',help='兼容性有问题，我太过时了，你自己换成transformer吧')


    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--mid_dim", type=int, default=268)
    parser.add_argument("--event_num", type=int, default=32)
    parser.add_argument("--argument_num", type=int, default=3)

    parser.add_argument("--do_ace_test",default=True)
    parser.add_argument("--do_event_test", default=True)
    parser.add_argument("--do_multi_event_test",default=False)
    args=parser.parse_args()
    return args





"""
class Config(object):
    #配置参数

    def __init__(self, dataset):
        self.model_name = 'bert'


        self.output_dir='result'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备
        self.train_path='data/train.json'
        self.dev_path='data/dev.json'
        self.test_path='data/test.json'

        self.num_epochs = 30  # epoch数
        self.batch_size = 64   # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate=3e-5
        self.weight_decay=0.01
        self.bert_path = '/home/rysong/work/pretrained_model/bert_base_chinese'
        self.token_nizer = BertTokenizer.from_pretrained(self.bert_path)
        # 加载bert的分词器
        self.hidden_dim = 768
        self.mid_dim = 268
        self.event_num = 32
        self.argument_num = 3
"""


