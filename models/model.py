# -*- coding: utf-8 -*-
"""
@Author : rysong
@software: pycharm
@File : model.py
@Time : 2021/9/19 6:29 下午
沉着冷静，逻辑清晰
"""
import torch
from pytorch_pretrained_bert import BertModel
from models.CRF import *


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))

        self.weight_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * weight + bias

        return outputs


class TriggerModel(nn.Module):
    def __init__(self, config):
        super(TriggerModel, self).__init__()
        self.config=config

        self.bert = BertModel.from_pretrained(config.bert_path)

        for param in self.bert.parameters():
            param.requires_grad = True   #对于模型的参数，使其反向传播，不固定模型参数

        # 需要这层么？？mid_dim=?
        #一层隐藏层+全连接
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim, config.mid_dim),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(config.mid_dim,config.event_num*2+1+2))
        kwargs = dict({'target_size': 2 * config.event_num + 1, 'device': config.device})
        self.CRF = CRF(**kwargs)

        # self.classifier = nn.Linear(self.mid_dim, self.event_num + 1)

    def forward(self, input_ids, attention_mask, labels,token_type_ids=None):
        bert_outputs = self.bert(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 output_all_encoded_layers=None)

        seq_out = bert_outputs[0]
        seq_out = self.fc(seq_out)
        loss = self.CRF.neg_log_likelihood_loss(seq_out, attention_mask, labels)
        _, out = self.CRF.forward(feats=seq_out,
                                       mask=attention_mask)
        return out, loss


class ArgumentModel(nn.Module):
    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):
        """
        实现类似 tf.batch_gather 的效果
        :param data: (bs, max_seq_len, hidden)
        :param index: (bs, n)   n=2,index=[start,end]
        :return: a tensor which shape is (bs, n, hidden)
        """
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)
        return torch.gather(data, 1, index)

    def __init__(self, config):
        super(ArgumentModel, self).__init__()
        self.config=config
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 需要这层么？？mid_dim=?
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim, config.mid_dim),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(config.mid_dim,config.argument_num*2+1+2))
        kwargs = dict({'target_size': 2 * config.argument_num + 1, 'device': self.config.device})
        self.CRF = CRF(**kwargs)
        out_dims = self.bert.config.hidden_size
        self.conditional_layer_norm = ConditionalLayerNorm(out_dims)

    def forward(self, input_ids, attention_mask, trigger_index, labels, token_type_ids=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=None
        )

        seq_out, pooled_out = bert_outputs[0], bert_outputs[1]

        trigger_label_feature = self._batch_gather(seq_out, trigger_index) #(bs,2,hidden)

        trigger_label_feature = trigger_label_feature.view(trigger_label_feature.size()[0], -1)
        # trigger_label_feature.shape=(batch_size,n*hidden)

        seq_out = self.conditional_layer_norm(seq_out, trigger_label_feature)

        seq_out = self.fc(seq_out)
        loss = self.CRF.neg_log_likelihood_loss(seq_out, attention_mask, labels)
        _,out = self.CRF.forward(seq_out, attention_mask)
        return out, loss
