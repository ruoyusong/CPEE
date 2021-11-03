# -*- coding: utf-8 -*-
"""
@Author : rysong
@software: pycharm
@File : train
@Time : 2021/9/23 11:30 上午 
沉着冷静，逻辑清晰
"""
import torch
from torch import nn
from models.model import TriggerModel, ArgumentModel
from utils.config import config
from datasets.dataset import TriggerDataset, ArgumentDataset, collate_fn_t, collate_fn_a
from torch.utils.data import DataLoader
from utils.evaluator import evaluate
from utils.function_utils import Logger, seed_everything
import os



def train(model, train_iter, config, task_type):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for i, batch in enumerate(train_iter):
        token, seq_len, input_ids, attention_mask = batch['all_token'], batch['all_seq_len'], batch['all_input_ids'].to(
            config.device), batch['all_attention_mask'].to(
            config.device)
        optimizer.zero_grad()
        if task_type == 'trigger':
            trigger_labels = batch["all_trigger_labels"].to(config.device)
            out, loss = model(input_ids, attention_mask, trigger_labels)
        else:
            trigger_index = batch["all_trigger_index"].to(config.device)
            argument_labels = batch["all_argument_labels"].to(config.device)
            out, loss = model(input_ids, attention_mask, trigger_index, argument_labels)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        print("step: {}, loss: {}".format(i, loss.item()))


if __name__ == '__main__':
    config = config()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device=device
    task_type=config.task_type
    seed_everything(11)


    if not os.path.exists(config.model_save_path):
        os.mkdir(config.model_save_path)
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("logs/dev"):
        os.mkdir("logs/dev")
    if not os.path.exists('logs/dev/{}'.format(task_type)):
        os.mkdir('logs/dev/{}'.format(task_type))

    logger = Logger('logs/dev/{}/{}+{}.log'.format(task_type, config.batch_size, config.learning_rate))
    logger.info("\n\n*******************This is a new dev log！batch_size:{},learning_rate:{}".format(config.batch_size,
                                                                                                 config.learning_rate))

    if task_type == 'trigger':
        train_dataset = TriggerDataset(config.train_path, config=config)
        dev_dataset = TriggerDataset(config.dev_path, config=config)
        model = TriggerModel(config).to(config.device)
        collate_fn = collate_fn_t
    else:
        train_dataset = ArgumentDataset(config.train_path, config)
        dev_dataset = ArgumentDataset(config.dev_path, config)
        model = ArgumentModel(config).to(config.device)
        collate_fn = collate_fn_a
    logger.info("训练数据有{}".format(len(train_dataset)))
    logger.info("验证数据有{}".format(len(dev_dataset)))

    print("训练数据有{}".format(len(train_dataset)))
    print("验证数据有{}".format(len(dev_dataset)))

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=config.batch_size,
                            collate_fn=collate_fn,
                            )
    dev_iter = DataLoader(dataset=dev_dataset,
                          batch_size=config.batch_size,
                          collate_fn=collate_fn)
    best_f1 = 0
    best_epoch = 0
    for epoch in range(config.num_epochs):
        print("epoch{}".format(epoch))
        logger.info("---------epoch{}--------".format(epoch))
        train(model, train_iter, config, task_type)
        p, r, f1 = evaluate(model, dev_iter, config, task_type)
        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save(model.state_dict(),os.path.join(config.model_save_path,"best_{}_model.pth".format(task_type)))#.pt/.pyh
    logger.info("The best {} model in epoch {}:{}".format(task_type, best_epoch, best_f1))
