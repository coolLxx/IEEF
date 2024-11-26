import random
import os

import torch
import numpy as np

from config import get_config
from runner import MultiWOZRunner, EDRunner
from utils.utils import get_or_create_logger

logger = get_or_create_logger(__name__)

def main():
    cfg = get_config()

    # main.py训练的参数列表: python main.py  -data_version 3.0 -agent_type us -run_type train -backbone t5-large -model_dir simulator_t5_large_data3.0 -epoch 10
    cfg.data_version = '3.0'
    cfg.agent_type = 'us'
    cfg.run_type = 'train'
    cfg.backbone = 't5-large'
    # cfg.aug_cutoff_ratio = 0.05
    cfg.learning_rate = 5e-4
    cfg.batch_size = 4
    cfg.epochs = 15
    # cfg.model_dir = "simulator/simulator_{}_data{}_lr{}_bs{}_cfr{}".format(cfg.backbone, cfg.data_version, cfg.learning_rate, cfg.batch_size, cfg.aug_cutoff_ratio)
    cfg.model_dir = "simulator/simulator_{}_data{}_lr{}_bs{}".format(cfg.backbone, cfg.data_version,
                                                                     cfg.learning_rate, cfg.batch_size)
    cfg.max_to_keep_ckpt = 20  # <MOD> 保存20个

    # main.py推理的参数列表: -data_version 3.0 -run_type predict -predict_agent_type us -pred_data_type test -ckpt ./interact_model/simulator_t5_large_data3.0_interact_09/ckpt-epoch10 -output inference.json -batch_size 4
    # cfg.data_version = '3.0'
    # cfg.run_type = 'predict'
    # cfg.predict_agent_type = 'us'
    # cfg.pred_data_type = 'test'
    # cfg.ckpt = './simulator/simulator_t5_large_data3.0/simulator_rl_dc0.99_lr1e-05_gc1_epoch_4'
    # cfg.output = 'inference.json'
    # cfg.batch_size = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

    setattr(cfg, "device", device)
    setattr(cfg, "num_gpus", num_gpus)

    logger.info("Device: %s (the number of GPUs: %d)", str(device), num_gpus)

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

    runner = EDRunner(cfg)  # runner = MultiWOZRunner(cfg)

    # # 编码后的数据格式
    # import pickle
    # with open('data/empathetic_dialogues/data_3.0/encoded_data.pkl', "rb") as f:
    #     data = pickle.load(f)
    # import json
    # with open('data/empathetic_dialogues/data_3.0/encoded_data.json', 'w') as f:
    #     json.dump(data, f)
    # exit()
    # import pickle
    # with open('data/MultiWOZ_2.0/processed/unencoded_demo.pkl', "rb") as f:
    #     data = pickle.load(f)
    # print(data)
    # import json
    # with open('data/MultiWOZ_2.0/processed/unencoded_demo.json', 'w') as f:
    #     json.dump(data, f)
    # exit()

    # # 整理成batch的数据格式
    # train_batches, _, _, _ = runner.iterator.get_batches("train", cfg.batch_size,
    #                                                      cfg.num_gpus, shuffle=True,
    #                                                      num_dialogs=10,  # cfg.num_train_dialogs,
    #                                                      excluded_domains=cfg.excluded_domains)
    # # print(train_batches)
    # import json
    # with open('data/empathetic_dialogues/data_3.0/train_demo_batches.json', 'w') as f:
    #     json.dump(train_batches, f)
    # exit()


    # 对话系统的训练数据和用户模拟器的训练数据的区别
    # get_data_iterator_ds = runner.iterator.get_data_iterator('ds')
    # get_data_iterator_us = runner.iterator.get_data_iterator('us')
    # train_ds_iterator = get_data_iterator_ds(train_batches, cfg.ururu, cfg.context_size)
    # train_us_iterator = get_data_iterator_us(train_batches, cfg.ururu, cfg.context_size)
    # # from transformers import T5Tokenizer
    # # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # print('--------------对话系统的训练数据-----------------')
    # for step, batch in enumerate(train_ds_iterator):
    #     inputs, resp_labels = batch
    #     print('inputs----------------')
    #     inputs = runner.reader.tokenizer.decode(inputs[0])
    #     print(inputs)
    #     print('resp_labels----------------')
    #     resp_labels = runner.reader.tokenizer.decode(resp_labels[0])
    #     print(resp_labels)
    # print('--------------用户模拟器的训练数据-----------------')
    # for step, batch in enumerate(train_us_iterator):
    #     inputs, resp_labels = batch
    #     print(inputs)
    #     print(resp_labels)
    #     print('inputs----------------')
    #     print(runner.reader.tokenizer.decode(inputs[0]))
    #     print(runner.reader.tokenizer.decode(inputs[1]))
    #     print(runner.reader.tokenizer.decode(inputs[2]))
    #     print(runner.reader.tokenizer.decode(inputs[3]))
    #     print('resp_labels----------------')
    #     resp_labels = runner.reader.tokenizer.decode(resp_labels[0])
    #     print(resp_labels)
    #     exit()
    # exit()


    if cfg.run_type == "train":
        runner.train()
    else:
        if cfg.predict_agent_type == 'ds':
            runner.predict()
        elif cfg.predict_agent_type == 'us':
            runner.us_predict()

if __name__ == "__main__":
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7891"
    # os.environ["HTTPS_PROXY"] = "https://127.0.0.1:7891"
    # os.environ["FTP_PROXY"] = "http://127.0.0.1:7891"
    # os.environ["ALL_PROXY"] = "http://127.0.0.1:7891"
    # os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    main()

