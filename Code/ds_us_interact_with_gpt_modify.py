import os
import spacy
import random
import json
import re
import copy
from abc import *
import difflib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
from itertools import chain
from utils.utils import get_or_create_logger, load_json, save_json, load_json_by_line, split_user_resp, split_system_resp, calculate_bleu
from transformers import T5ForConditionalGeneration, T5Tokenizer, BlenderbotSmallForConditionalGeneration, AutoTokenizer
from openai import OpenAI
from utils import definitions
from collections import OrderedDict
# from reader import BaseReader
# from runner import EDRunner
from runner import Reporter
from config import get_config

from transformers import get_linear_schedule_with_warmup, get_constant_schedule  # AdamW
from torch.optim import AdamW
import glob
import shutil
import time

logger = get_or_create_logger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # user
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # ds


def inter_get_config():
    parser = argparse.ArgumentParser(description='ds us interact with gpt modify')
    parser.add_argument("-simulator_path", type=str,
                        default='./simulator/simulator_t5_large_data3.0/ckpt-epoch10')  #  ./simulator_t5_large_data3.0/ckpt-epoch10   ./simulator_t5_small/ckpt-epoch4  ./simulator_t5_large_lr5e-4_bs4/ckpt-epoch5  ./simulator_t5_large_data3.0/ckpt-epoch10
    parser.add_argument("-dialog_sys_path", type=str,
                        default='./dialogue/dialogue_t5_large_data3.0/ckpt-epoch8')  # ./dialogue_t5_large_data3.0/ckpt-epoch8 ./dialogue_t5_small/ckpt-epoch6  ./dialogue_t5_large_lr5e-4_bs2/ckpt-epoch5  ./dialogue_t5_large_data3.0/ckpt-epoch10
    parser.add_argument('-us_model_name', type=str, default='t5-large', choices=['t5-small', 't5-large'])
    parser.add_argument('-ds_model_name', type=str, default='t5-large', choices=['t5-small', 't5-large', 'blenderbot_small'])
    parser.add_argument("-max_turn_num", type=int, default=5)
    # parser.add_argument("-data_path", type=str, default='./data/empathetic_dialogues/ieval_data.json')
    parser.add_argument("-data_dir", type=str, default='./data/empathetic_dialogues')
    parser.add_argument('-data_version', type=str, default='3.0', choices=['1.0', '2.0', '3.0'],
                        help="1.0: 原始版本 2.0: 续写版本 3.0: 续写简短版本")
    parser.add_argument('-generate_results_path', type=str,
                        default='interact_output/t5_large_generate_modify/tmp')  # output.json
    parser.add_argument('-interaction_type', type=str, default='train', choices=['train', 'test', 'dev'])
    args = parser.parse_args()

    return args


def export_api_key():
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7896'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7896'
    os.environ['FTP_PROXY'] = 'http://127.0.0.1:7896'
    os.environ['ALL_PROXY'] = 'http://127.0.0.1:7896'
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    os.environ['OPENAI_API_KEY'] = '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'



class BaseIterator(object):
    def __init__(self, reader):
        self.reader = reader

    def bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []

            turn_bucket[turn_len].append(dial)

        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def construct_mini_batch(self, data, batch_size, num_gpus):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []

        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if (len(batch) % num_gpus) != 0:
            batch = batch[:-(len(batch) % num_gpus)]
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)

        return all_batches

    def transpose_batch(self, dial_batch):
        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

    def get_batches(self, data_type, batch_size, num_gpus, shuffle=False, num_dialogs=-1, excluded_domains=None):
        dial = self.reader.data[data_type]

        if num_dialogs > 0:
            dial = random.sample(dial, min(num_dialogs, len(dial)))

        turn_bucket = self.bucket_by_turn(dial)

        all_batches = []

        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        for k in turn_bucket:
            if data_type != "test" and (k == 1 or k >= 17):
                continue

            batches = self.construct_mini_batch(
                turn_bucket[k], batch_size, num_gpus)

            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches

        if shuffle:
            random.shuffle(all_batches)

        return all_batches, num_training_steps, num_dials, num_turns

    def flatten_dial_history(self, dial_history, len_postfix, context_size=-1):
        if context_size > 0:
            context_size -= 1

        if context_size == 0:
            windowed_context = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
        else:
            windowed_context = dial_history

        ctx_len = sum([len(c) for c in windowed_context])

        # consider eos_token
        spare_len = self.reader.max_seq_len - len_postfix - 1
        while ctx_len >= spare_len:
            ctx_len -= len(windowed_context[0])
            windowed_context.pop(0)

        context = list(chain(*windowed_context))

        return context

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long)

    def get_data_iterator(self, all_batches, task, ururu, add_auxiliary_task=False, context_size=-1):
        raise NotImplementedError


class EDIterator(BaseIterator):
    def __init__(self, reader):
        super(EDIterator, self).__init__(reader)

    def get_readable_batch(self, dial_batch):
        dialogs = {}

        decoded_keys = ["user", "resp", "redx", "bspn", "aspn", "dbpn",
                        "bspn_gen", "bspn_gen_with_span",
                        "dbpn_gen", "aspn_gen", "resp_gen", "user_aspn", "goal_state"]
        for dial in dial_batch:
            dial_id = dial[0]["dial_id"]

            dialogs[dial_id] = []

            for turn in dial:
                readable_turn = {}

                for k, v in turn.items():
                    if k in ["dial_id", "resp_span", "user_span"]:
                        continue
                    elif k in decoded_keys:
                        v = self.reader.tokenizer.decode(
                            v, clean_up_tokenization_spaces=False)
                        '''
                        if k == "user":
                            print(k, v)
                        '''
                    elif k == "pointer":
                        turn_doamin = turn["turn_domain"][-1]
                        v = self.reader.db.pointerBack(v, turn_doamin)
                    if k == "user_span" or k == "resp_span":
                        speaker = k.split("_")[0]
                        v_dict = {}
                        for domain, ss_dict in v.items():
                            v_dict[domain] = {}
                            for s, span in ss_dict.items():
                                v_dict[domain][s] = self.reader.tokenizer.decode(
                                    turn[speaker][span[0]: span[1]])
                        v = v_dict

                    readable_turn[k] = v

                dialogs[dial_id].append(readable_turn)

        return dialogs

    def get_data_iterator(self, training_type):
        # pick particular data iterator
        if training_type == 'ds':
            return self.get_data_iterator_ds
        elif training_type == 'us':
            return self.get_data_iterator_us

    def get_data_iterator_us(self, all_batches, ururu, context_size=-1):
        # data iterator for trianing user simulator
        for dial_batch in all_batches:
            batch_encoder_input_ids = []
            batch_resp_label_ids = []  # user action labels and user utterance labels

            for dial in dial_batch:
                dial_encoder_input_ids = []
                dial_resp_label_ids = []

                dial_history = []
                for turn in dial:
                    context = self.flatten_dial_history(
                        dial_history, len(turn['goal_state']), context_size
                    )
                    encoder_input_ids = context + turn['goal_state'] + [self.reader.eos_token_id]
                    resp = turn['user']  # turn['user_aspn'] + turn['user']
                    resp_label_ids = resp + [self.reader.eos_token_id]

                    dial_encoder_input_ids.append(encoder_input_ids)
                    dial_resp_label_ids.append(resp_label_ids)

                    # if ururu:
                    #     turn_text = turn['user'] + turn['redx']
                    # else:
                    #     turn_text = turn['user'] + turn['bspn'] + turn['dbpn'] + turn['aspn'] + turn['redx']

                    turn_text = turn['user'] + turn['resp']

                    dial_history.append(turn_text)

                batch_encoder_input_ids.append(dial_encoder_input_ids)
                batch_resp_label_ids.append(dial_resp_label_ids)

            # turn first
            batch_encoder_input_ids = self.transpose_batch(batch_encoder_input_ids)
            batch_resp_label_ids = self.transpose_batch(batch_resp_label_ids)

            num_turns = len(batch_encoder_input_ids)

            tensor_encoder_input_ids = []
            tensor_resp_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids = [self.tensorize(b) for b in batch_encoder_input_ids[t]]
                tensor_resp_label_ids = [self.tensorize(b) for b in batch_resp_label_ids[t]]
                tensor_encoder_input_ids = pad_sequence(tensor_encoder_input_ids, batch_first=True,
                                                        padding_value=self.reader.pad_token_id)
                tensor_resp_label_ids = pad_sequence(tensor_resp_label_ids, batch_first=True,
                                                     padding_value=self.reader.pad_token_id)

                yield tensor_encoder_input_ids, tensor_resp_label_ids  # , None

    def get_data_iterator_ds(self, all_batches, ururu, context_size=-1):
        # data iterator for training dialogue system
        for dial_batch in all_batches:
            batch_encoder_input_ids = []
            # batch_belief_label_ids = [] # <MOD>
            batch_resp_label_ids = []

            for dial in dial_batch:
                dial_encoder_input_ids = []
                # dial_belief_label_ids = []
                dial_resp_label_ids = []

                dial_history = []
                for turn in dial:
                    context = self.flatten_dial_history(
                        dial_history, len(turn["user"]), context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    # bspn = turn["bspn"]

                    # bspn_label = bspn

                    # belief_label_ids = bspn_label + [self.reader.eos_token_id]
                    resp = turn['resp']  # turn['dbpn'] + turn["aspn"] + turn["redx"]
                    resp_label_ids = resp + [self.reader.eos_token_id]

                    dial_encoder_input_ids.append(encoder_input_ids)
                    # dial_belief_label_ids.append(belief_label_ids)
                    dial_resp_label_ids.append(resp_label_ids)

                    # if ururu:
                    #     turn_text = turn["user"] + turn["redx"]
                    # else:
                    #     turn_text = turn["user"] + bspn + turn["dbpn"] + turn["aspn"] + turn["redx"]
                    turn_text = turn['user'] + turn['resp']

                    dial_history.append(turn_text)

                batch_encoder_input_ids.append(dial_encoder_input_ids)
                # batch_belief_label_ids.append(dial_belief_label_ids)
                batch_resp_label_ids.append(dial_resp_label_ids)

            # turn first
            batch_encoder_input_ids = self.transpose_batch(batch_encoder_input_ids)
            # batch_belief_label_ids = self.transpose_batch(batch_belief_label_ids)
            batch_resp_label_ids = self.transpose_batch(batch_resp_label_ids)

            num_turns = len(batch_encoder_input_ids)

            # tensor_encoder_input_ids = []
            # tensor_belief_label_ids = []
            # tensor_resp_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids = [
                    self.tensorize(b) for b in batch_encoder_input_ids[t]]
                # tensor_belief_label_ids = [
                #     self.tensorize(b) for b in batch_belief_label_ids[t]]
                tensor_resp_label_ids = [
                    self.tensorize(b) for b in batch_resp_label_ids[t]]

                tensor_encoder_input_ids = pad_sequence(tensor_encoder_input_ids,
                                                        batch_first=True,
                                                        padding_value=self.reader.pad_token_id)
                # tensor_belief_label_ids = pad_sequence(tensor_belief_label_ids,
                #                                        batch_first=True,
                #                                        padding_value=self.reader.pad_token_id)

                tensor_resp_label_ids = pad_sequence(tensor_resp_label_ids,
                                                     batch_first=True,
                                                     padding_value=self.reader.pad_token_id)

                yield tensor_encoder_input_ids, tensor_resp_label_ids  # , tensor_belief_label_ids



class BaseReader(object):
    def __init__(self, tokenizer):  # (self, cfg):
        # self.cfg = cfg
        self.nlp = spacy.load("en_core_web_sm")

        self.tokenizer = tokenizer

        self.data_dir = self.get_data_dir()

        train = self.encode_data("train")
        dev = self.encode_data("valid")
        test = self.encode_data("test")
        self.data = {"train": train, "dev": dev, "test": test}

        # encoded_data_path = os.path.join(self.data_dir, "encoded_data.pkl")  # <MOD>
        # # encoded_data_path = os.path.join(self.data_dir, "encoded_data_{}.pkl".format(cfg.model_name))
        #
        # if os.path.exists(encoded_data_path):
        #     logger.info("Load encoded data from {}".format(encoded_data_path))
        #
        #     self.data = load_pickle(encoded_data_path)
        #
        # else:
        #     logger.info("Encode data and save to {}".format(encoded_data_path))
        #     train = self.encode_data("train")
        #     dev = self.encode_data("valid")  # <MOD>
        #     test = self.encode_data("test")
        #
        #     self.data = {"train": train, "dev": dev, "test": test}
        #
        #     save_pickle(self.data, encoded_data_path)

    def get_data_dir(self):
        raise NotImplementedError

    # # <MOD>
    # def init_tokenizer(self):
    #     if self.cfg.ckpt is not None:
    #         return T5Tokenizer.from_pretrained(self.cfg.ckpt)
    #     elif self.cfg.train_from is not None:
    #         return T5Tokenizer.from_pretrained(self.cfg.train_from)
    #     else:
    #         tokenizer = T5Tokenizer.from_pretrained(self.cfg.backbone)
    #
    #     special_tokens = []
    #
    #     special_tokens.append("<bos_user>")
    #     special_tokens.append("<eos_user>")
    #     special_tokens.append("<bos_resp>")
    #     special_tokens.append("<eos_resp>")
    #     special_tokens.append("<bos_goal>")
    #     special_tokens.append("<eos_goal>")
    #     special_tokens.append("[emotion]")
    #     special_tokens.append("[situation]")
    #
    #     tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    #
    #     return tokenizer

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text

        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]

            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]

            tokens = tokens + eos_token

        # <MOD>
        # encoded_text = tokens
        encoded_text = self.tokenizer.encode(" ".join(tokens))

        # except eos token
        if encoded_text[-1] == self.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def encode_data(self, data_type):
        raise NotImplementedError


class EDReader(BaseReader):
    def __init__(self, tokenizer, epoch):  # (self, cfg):
        self.epoch = epoch  # <MOD>
        super(EDReader, self).__init__(tokenizer)

    # data/empathetic_dialogues/original_data
    def get_data_dir(self):
        # return os.path.join(
        #     "data", "MultiWOZ_{}".format(self.version), "processed")
        # if self.cfg.data_version == '1.0':
        #     return "data/empathetic_dialogues/original_data"
        # elif self.cfg.data_version == '2.0':
        #     return "data/empathetic_dialogues/datagpt3_5"
        return "./interact_output/t5_large_generate_modify/train"

    def encode_data(self, data_type):
        # data_path = ''  # <MOD>
        # if self.cfg.data_version == '1.0':
        #     data_path = os.path.join(self.data_dir, "{}.jsonl".format(data_type))
        # elif self.cfg.data_version == '2.0':
        #     data_path = os.path.join(self.data_dir, "{}_gpt3_5.jsonl".format(data_type))
        if data_type == "train":
            data_path = os.path.join(self.data_dir, "train_%03d.json" % self.epoch)    # <NEED> <MOD>
            logger.info("Load train data from {}".format(data_path))
        elif data_type == "valid":
            data_path = "./data/empathetic_dialogues/data_3.0/valid.jsonl"    # <NEED> <MOD>
            logger.info("Load valid data from {}".format(data_path))
        elif data_type == "test":
            data_path = "./data/empathetic_dialogues/data_3.0/test.jsonl"
            logger.info("Load test data from {}".format(data_path))

        data = load_json_by_line(data_path)

        encoded_data = []
        convs = []
        dial = []

        # 把每个dial整合到一起
        for idx, line in enumerate(data):
            dial.append(line)
            # 清空前保存dial
            if idx + 1 == len(data) or line['conv_idx'] != data[idx + 1]['conv_idx']:
                convs.append(dial)
                dial = []

        # 定义特殊token
        bos_user_token = '<bos_user>'
        eos_user_token = '<eos_user>'

        bos_resp_token = '<bos_resp>'
        eos_resp_token = '<eos_resp>'

        bos_goal_token = '<bos_goal>'
        eos_goal_token = '<eos_goal>'

        # 调整格式并编码
        new_convs = []
        for dial_id, dial in enumerate(convs):
            new_dial = []
            len_dial = len(dial)
            # 如果不是偶数个对话语句的时候去掉最后一句
            if len_dial % 2 != 0:
                len_dial -= 1

            for idx in range(0, len_dial, 2):
                # 把两句话（一轮对话）整合到一起
                turn = {}
                turn['dial_id'] = data_type + "_" + str(dial_id)
                turn['turn_num'] = int(idx / 2)
                turn['user'] = self.encode_text(dial[idx]['utterance'], bos_token=bos_user_token, eos_token=eos_user_token)
                turn['resp'] = self.encode_text(dial[idx + 1]['utterance'], bos_token=bos_resp_token,
                                           eos_token=eos_resp_token)
                turn['goal_state'] = "[emotion] " + dial[idx]['emotion'] + " [situation] " + dial[idx]['situation']
                turn['goal_state'] = self.encode_text(turn['goal_state'], bos_token=bos_goal_token, eos_token=eos_goal_token)
                new_dial.append(turn)
            new_convs.append(new_dial)

        return new_convs


class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg
        self.reader = None
        self.iterator = None
        self.model = None

    def save_model(self, ckpt_name):
        save_path = os.path.join(self.cfg.model_dir, ckpt_name)
        model = self.model
        model.save_pretrained(save_path)
        self.reader.tokenizer.save_pretrained(save_path)

        logger.info("Save model in {}".format(save_path))


    # def save_model(self, epoch):
    #     latest_ckpt = "ckpt-epoch{}".format(epoch)
    #     save_path = os.path.join(self.cfg.model_dir, latest_ckpt)
    #     '''
    #     if self.cfg.num_gpus > 1:
    #         model = self.model.module
    #     else:
    #         model = self.model
    #     '''
    #     model = self.model
    #     model.save_pretrained(save_path)
    #     self.reader.tokenizer.save_pretrained(save_path)
    #
    #     # keep chekpoint up to maximum
    #     checkpoints = sorted(
    #         glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
    #         key=os.path.getmtime,
    #         reverse=True)
    #
    #     checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]
    #
    #     for ckpt in checkpoints_to_be_deleted:
    #         shutil.rmtree(ckpt)
    #
    #     return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_traininig_steps_per_epoch *
            self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            #num_warmup_steps = int(num_train_steps * 0.2)
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        num_count = label.view(-1).ne(pad_id).long().sum()
        num_correct = 0
        for i in range(label.shape[0]):
            single_pred, single_label = pred[i], label[i]
            valid_len = single_label.ne(pad_id).long().sum()
            single_pred = single_pred[:valid_len]
            single_label = single_label[:valid_len]
            num_correct += (single_pred == single_label).sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

class EDRunner(BaseRunner):
    def __init__(self, cfg):
        # reader = EDReader(cfg)  # MultiWOZReader(cfg, cfg.version)
        #
        # self.iterator = EDIterator(reader)  # MultiWOZIterator(reader)

        super(EDRunner, self).__init__(cfg)

    # <MOD> 添加数据增强方法
    def generate_token_cutoff_embedding(self, embeds, input_lens, aug_cutoff_ratio):
        input_embeds = []
        for i in range(embeds.shape[0]):
            cutoff_length = int(input_lens * aug_cutoff_ratio)
            zero_index = torch.randint(input_lens, (cutoff_length,))

            cutoff_embed = embeds[i]
            tmp_mask = torch.ones(cutoff_embed.shape[0], ).to(embeds.device)
            for ind in zero_index:
                tmp_mask[ind] = 0

            cutoff_embed = torch.mul(tmp_mask[:, None], cutoff_embed)

            input_embeds.append(cutoff_embed)

        input_embeds = torch.stack(input_embeds, dim=0)

        return input_embeds

    def generate_span_cutoff_embedding(self, embeds, input_lens, aug_cutoff_ratio):
        input_embeds = []
        for i in range(embeds.shape[0]):
            cutoff_length = int(input_lens * aug_cutoff_ratio)
            start = int(torch.rand(1) * (input_lens - cutoff_length))
            # print(input_lens[i], cutoff_length, start)
            cutoff_embed = torch.cat((embeds[i][:start],
                                      torch.zeros([cutoff_length, embeds.shape[-1]],
                                                  dtype=torch.float).to(embeds.device),
                                      embeds[i][start + cutoff_length:]), dim=0)
            input_embeds.append(cutoff_embed)
        input_embeds = torch.stack(input_embeds, dim=0)
        return input_embeds


    def step_fn(self, inputs, resp_labels):  # , belief_labels=None):
        '''
        修改：增加数据增强  --已撤销
        '''
        inputs = inputs.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)
        # if self.cfg.agent_type == 'ds' and belief_labels is not None:
        #     belief_labels = belief_labels.to(self.cfg.device)

        attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)

        # inputs_embeds = self.model.shared(inputs)  # <MOD>
        # _, seq_len, _ = inputs_embeds.size()  # <MOD>
        # inputs_embeds = self.generate_span_cutoff_embedding(inputs_embeds, seq_len, aug_cutoff_ratio=self.cfg.aug_cutoff_ratio)  # <MOD>

        if self.cfg.agent_type == 'ds':
            resp_outputs = self.model(input_ids=inputs,  # inputs_embeds=inputs_embeds,  <MOD>
                                      attention_mask=attention_mask,
                                      labels=resp_labels)
            resp_loss = resp_outputs.loss
            resp_logits = resp_outputs.logits
            resp_pred = torch.argmax(resp_logits, dim=-1)
            num_resp_correct, num_resp_count = self.count_tokens(resp_pred, resp_labels,
                                                                 pad_id=self.reader.pad_token_id)
        elif self.cfg.agent_type == 'us':
            resp_outputs = self.model(input_ids=inputs,  # inputs_embeds=inputs_embeds,  <MOD>
                                      attention_mask=attention_mask,
                                      labels=resp_labels)
            resp_loss = resp_outputs.loss
            resp_logits = resp_outputs.logits
            resp_pred = torch.argmax(resp_logits, dim=-1)
            num_resp_correct, num_resp_count = self.count_tokens(resp_pred, resp_labels,
                                                                 pad_id=self.reader.pad_token_id)
        else:
            raise Exception('Wrong agent type! It should be us or ds.')

        loss = self.cfg.resp_loss_coeff * resp_loss

        # if self.cfg.agent_type == 'ds' and belief_labels is not None:
        #     loss += self.cfg.bspn_loss_coeff * belief_loss

        step_outputs = {}
        step_outputs["resp"] = {"loss": resp_loss.item(),
                                "correct": num_resp_correct.item(),
                                "count": num_resp_count.item()}

        return loss, step_outputs

    def train_epoch(self, train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter=None, epoch=None):
        self.model.train()
        self.model.zero_grad()

        with tqdm(total=num_training_steps_per_epoch, desc="Epoch {}".format(epoch)) as pbar:  # <MOD> 增加了desc
            for step, batch in enumerate(train_iterator):
                start_time = time.time()

                inputs, resp_labels = batch  # inputs, resp_labels, belief_labels = batch

                loss, step_outputs = self.step_fn(inputs, resp_labels)  # loss, step_outputs = self.step_fn(inputs, resp_labels, belief_labels)

                if self.cfg.grad_accum_steps > 1:
                    loss = loss / self.cfg.grad_accum_steps

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm)

                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    lr = scheduler.get_last_lr()[0]

                    if reporter is not None and self.cfg.log_frequency > 0:
                        reporter.step(start_time, lr, step_outputs)
                pbar.update(1)

    def train(self):
        train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
            "train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True,
            num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size)

        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        # bestbleu_ckpt, bestloss_ckpt, latest_ckpt = self.model, self.model, self.model
        # bestbleu, bestloss = 0, 1e8

        for epoch in range(1, self.cfg.epochs + 1):  # <MOD>
            get_iterator_fn = self.iterator.get_data_iterator(self.cfg.agent_type)
            train_iterator = get_iterator_fn(train_batches, self.cfg.ururu, self.cfg.context_size)

            self.train_epoch(train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter, epoch)

            logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            # <MOD>
            if not self.cfg.no_validation:
                loss, bleu4 = self.validation(reporter.global_step, num_dialogs=-1)  # self.validation(reporter.global_step) 200指的是测试bleu指标的dev数据量
                if self.cfg.bestbleu4 < bleu4:
                    self.cfg.bestbleu4 = bleu4
                    self.save_model("bestbleu4_ckpt")
                if self.cfg.bestloss > loss:
                    self.cfg.bestbleu4 = loss
                    self.save_model("bestloss_ckpt")
                if epoch == self.cfg.epochs:
                    self.save_model("latest_ckpt")

            logger.info(" ".join(["[Validation]", "Best BLEU-4 Score: {:.2f}  Best Loss: {:.2f}".format(self.cfg.bestbleu4, self.cfg.bestloss)]))
            # self.save_model(epoch)  # <MOD>
        # self.save_model(self.cfg.epochs)  # <MOD>

    def validation(self, global_step, num_dialogs=-1):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)  # <MOD>  , num_dialogs=10

        get_iterator_fn = self.iterator.get_data_iterator(self.cfg.agent_type)

        dev_iterator = get_iterator_fn(
            dev_batches, self.cfg.ururu, self.cfg.context_size)

        reporter = Reporter(1000000, self.cfg.model_dir)

        torch.set_grad_enabled(False)
        all_loss = 0
        for batch in tqdm(dev_iterator, total=num_steps, desc="Validaction"):
            start_time = time.time()

            inputs, resp_labels = batch  # inputs, resp_labels, belief_labels = batch

            loss, step_outputs = self.step_fn(inputs, resp_labels)  # _, step_outputs = self.step_fn(inputs, resp_labels, belief_labels)
            all_loss += loss

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_belief_stats = True if 'belief' in step_outputs else False

        logger.info(" ".join(["[Validation]", "Loss: {:.2f}".format(all_loss)]))
        reporter.info_stats("dev", global_step, do_belief_stats)

        # <MOD>
        torch.cuda.empty_cache()
        origin_batch_size = self.cfg.batch_size
        self.cfg.batch_size = 32  # <MOD>
        if self.cfg.agent_type == 'ds':
            results = self.predict(num_dialogs)
            bleu4 = calculate_bleu(results, self.cfg.agent_type)
            reporter.info_bleu("dev", bleu4, global_step)
        elif self.cfg.agent_type == 'us':
            results = self.us_predict(num_dialogs)
            bleu4 = calculate_bleu(results, self.cfg.agent_type)
            reporter.info_bleu("dev", bleu4, global_step)
        self.cfg.batch_size = origin_batch_size
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)

        return all_loss, bleu4

    def finalize_bspn(self, belief_outputs, domain_history, constraint_history, span_outputs=None, input_ids=None):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn

            # update bspn using span output
            if span_outputs is not None and input_ids is not None:
                span_output = span_outputs[i]
                input_id = input_ids[i]

                # print(self.reader.tokenizer.decode(input_id))
                # print(self.reader.tokenizer.decode(bspn))

                eos_idx = input_id.index(self.reader.eos_token_id)
                input_id = input_id[:eos_idx]

                span_result = {}

                bos_user_id = self.reader.get_token_id(definitions.BOS_USER_TOKEN)

                span_output = span_output[:eos_idx]

                b_slot = None
                for t, span_token_idx in enumerate(span_output):
                    turn_id = max(input_id[:t].count(bos_user_id) - 1, 0)
                    turn_domain = domain_history[i][turn_id]

                    if turn_domain not in definitions.INFORMABLE_SLOTS:
                        continue

                    span_token = self.reader.span_tokens[span_token_idx]

                    if span_token not in definitions.INFORMABLE_SLOTS[turn_domain]:
                        b_slot = span_token
                        continue

                    if turn_domain not in span_result:
                        span_result[turn_domain] = defaultdict(list)

                    if b_slot != span_token:
                        span_result[turn_domain][span_token] = [input_id[t]]
                    else:
                        span_result[turn_domain][span_token].append(input_id[t])

                    b_slot = span_token

                for domain, sv_dict in span_result.items():
                    for s, v_list in sv_dict.items():
                        value = v_list[-1]
                        span_result[domain][s] = self.reader.tokenizer.decode(
                            value, clean_up_tokenization_spaces=False)

                span_dict = copy.deepcopy(span_result)

                ontology = self.reader.db.extractive_ontology

                flatten_span = []
                for domain, sv_dict in span_result.items():
                    flatten_span.append("[" + domain + "]")

                    for s, v in sv_dict.items():
                        if domain in ontology and s in ontology[domain]:
                            if v not in ontology[domain][s]:
                                del span_dict[domain][s]
                                continue

                        if s == "destination" or s == "departure":
                            _s = "destination" if s == "departure" else "departure"

                            if _s in sv_dict and v == sv_dict[_s]:
                                if s in span_dict[domain]:
                                    del span_dict[domain][s]
                                if _s in span_dict[domain]:
                                    del span_dict[domain][_s]
                                continue

                        if s in ["time", "leave", "arrive"]:
                            v = v.replace(".", ":")
                            if re.match("[0-9]+:[0-9]+", v) is None:
                                del span_dict[domain][s]
                                continue
                            else:
                                span_dict[domain][s] = v

                        flatten_span.append("[value_" + s + "]")
                        flatten_span.append(v)

                    if len(span_dict[domain]) == 0:
                        del span_dict[domain]
                        flatten_span.pop()

                # print(flatten_span)

                # input()

                decoded["span"] = flatten_span

                constraint_dict = self.reader.bspn_to_constraint_dict(
                    self.reader.tokenizer.decode(bspn, clean_up_tokenization_spaces=False))

                if self.cfg.overwrite_with_span:
                    _constraint_dict = OrderedDict()

                    for domain, slots in definitions.INFORMABLE_SLOTS.items():
                        if domain in constraint_dict or domain in span_dict:
                            _constraint_dict[domain] = OrderedDict()

                        for slot in slots:
                            if domain in constraint_dict:
                                cons_value = constraint_dict[domain].get(slot, None)
                            else:
                                cons_value = None

                            if domain in span_dict:
                                span_value = span_dict[domain].get(slot, None)
                            else:
                                span_value = None

                            if cons_value is None and span_value is None:
                                continue

                            # priority: span_value > cons_value
                            slot_value = span_value or cons_value

                            _constraint_dict[domain][slot] = slot_value
                else:
                    _constraint_dict = copy.deepcopy(constraint_dict)

                bspn_gen_with_span = self.reader.constraint_dict_to_bspn(
                    _constraint_dict)

                bspn_gen_with_span = self.reader.encode_text(
                    bspn_gen_with_span,
                    bos_token=definitions.BOS_BELIEF_TOKEN,
                    eos_token=definitions.EOS_BELIEF_TOKEN)

                decoded["bspn_gen_with_span"] = bspn_gen_with_span

            batch_decoded.append(decoded)

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                # logger.warn("bos/eos action token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                # logger.warn("bos/eos resp token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def predict(self, num_dialogs=-1):
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains, num_dialogs=num_dialogs)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn['user']), self.cfg.context_size
                    )

                    encoder_input_ids = context + turn['user'] + [self.reader.eos_token_id]
                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)
                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)
                attention_mask = torch.where(batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                with torch.no_grad():
                    model_outputs = self.model.generate(
                        input_ids=batch_encoder_input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=self.reader.eos_token_id,
                        max_length=200
                    )

                model_outputs = model_outputs.cpu().numpy().tolist()

                for t, turn in enumerate(turn_batch):
                    system_resp = split_system_resp(self.reader.tokenizer, model_outputs[t])
                    system_resp = self.reader.tokenizer.decode(system_resp,
                                                                  clean_up_tokenization_spaces=False).split()
                    system_resp = ' '.join(system_resp[1:-1])
                    turn['sys_gen'] = system_resp

                    pv_text = copy.copy(turn['user'])
                    pv_text = pv_text + turn['resp']
                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        return results


    def us_predict(self, num_dialogs=-1):
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains, num_dialogs=num_dialogs)

        # eval_dial_list = None
        # if self.cfg.excluded_domains is not None:
        #     eval_dial_list = []
        #
        #     for domains, dial_ids in self.iterator.dial_by_domain.items():
        #         domain_list = domains.split("-")
        #
        #         if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
        #             eval_dial_list.extend(dial_ids)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn['goal_state']), self.cfg.context_size
                    )

                    encoder_input_ids = context + turn['goal_state'] + [self.reader.eos_token_id]
                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)
                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)
                attention_mask = torch.where(batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                with torch.no_grad():
                    model_outputs = self.model.generate(
                        input_ids=batch_encoder_input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=self.reader.eos_token_id,
                        max_length=200
                    )

                model_outputs = model_outputs.cpu().numpy().tolist()

                for t, turn in enumerate(turn_batch):
                    # user_act, user_utterance, _, _ = split_user_act_and_resp(self.reader.tokenizer, model_outputs[t])
                    # user_act = self.reader.tokenizer.decode(user_act, clean_up_tokenization_spaces=False).split()
                    # user_utterance = self.reader.tokenizer.decode(user_utterance,
                    #                                               clean_up_tokenization_spaces=False).split()
                    # user_act = ' '.join(user_act[1:-1])
                    # user_utterance = ' '.join(user_utterance[1:-1])
                    # turn['user_gen'] = user_utterance
                    # turn['user_act_gen'] = user_act

                    user_utterance = split_user_resp(self.reader.tokenizer, model_outputs[t])
                    user_utterance = self.reader.tokenizer.decode(user_utterance,
                                                                  clean_up_tokenization_spaces=False).split()
                    user_utterance = ' '.join(user_utterance[1:-1])
                    turn['user_gen'] = user_utterance

                    pv_text = copy.copy(turn['user'])
                    pv_text = pv_text + turn['resp']
                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        # evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)
        # bleu = evaluator.e2e_eval(results, eval_for_us=True)
        # logger.info('bleu: {:2.2f}'.format(bleu))

        return results


class InteractionEnvironment(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.simulator_model = self.load_simulator(self.cfg.simulator_path)
        self.dialog_model = self.load_system(self.cfg.dialog_sys_path)

        self.simulator_model.to(device)
        self.dialog_model.to(device2)

        self.simulator_tokenizer = self.load_simulator_tokenizer(self.cfg.simulator_path)
        self.dialog_tokenizer = self.load_sys_tokenizer(self.cfg.dialog_sys_path)
        # self.data_path = self.cfg.data_path
        self.data_dir = os.path.join(cfg.data_dir, "data_{}".format(cfg.data_version))
        self.goal_list = self.get_goal_list()

    def load_simulator(self, model_path):
        logger.info("Load simulator model from {}".format(model_path))
        if not os.path.exists(model_path):
            raise Exception('Simulator Model path is invalid!')
        if self.cfg.us_model_name == "t5-large" or self.cfg.us_model_name == "t5-small":
            return T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            raise Exception('US Model name is invalid!')

    def load_system(self, model_path):
        logger.info("Load system model from {}".format(model_path))
        if not os.path.exists(model_path):
            raise Exception('System Model path is invalid!')
        if self.cfg.ds_model_name == "t5-large" or self.cfg.ds_model_name == "t5-small":
            return T5ForConditionalGeneration.from_pretrained(model_path)
        elif self.cfg.ds_model_name == "blenderbot_small":
            return BlenderbotSmallForConditionalGeneration.from_pretrained(model_path)
        else:
            raise Exception('DS Model name is invalid!')

    def load_simulator_tokenizer(self, tokenizer_path):
        logger.info("Load simulator tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('Simulator Tokenizer path is invalid!')
        if self.cfg.us_model_name == "t5-large" or self.cfg.us_model_name == "t5-small":
            return T5Tokenizer.from_pretrained(tokenizer_path)
        else:
            raise Exception('US Model name is invalid!')

    def load_sys_tokenizer(self, tokenizer_path):
        logger.info("Load system tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('System Tokenizer path is invalid!')
        if self.cfg.ds_model_name == "t5-large" or self.cfg.ds_model_name == "t5-small":
            return T5Tokenizer.from_pretrained(tokenizer_path)
        elif self.cfg.ds_model_name == "blenderbot_small":
            return AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            raise Exception('DS Model name is invalid!')

    def OpenAI_API_modify(self, dial_gen):
        log = dial_gen['log']
        goal = dial_gen['goal'].split()
        emotion = goal[1]
        situation = ' '.join(goal[3:])

        prompt = f'''I will give you a dialogue between a Speaker and a Listener. The Speaker will share his/her emotion and situation with the Listener, hoping that the Listener can understand and comfort him/her. Please help me analyze whether there are any inappropriate statements in the Speaker, such as incorrect syntax or semantic errors. If any errors are found in the Speaker's statements (ignore case), please provide me with the correct version with the minimum modification.

Please organize your reply in the following format:
[Whether modified] Choose between yes and no
[Reason] Your reason
[Modified conversation]
- [Modified Speaker] xxxx
- [Modified Listener] xxxx

Here is their dialogue:
'''

        for idx in range(self.cfg.max_turn_num):
            prompt += "[Speaker]: {}\n".format(log[idx]['user'])
            prompt += "[Listener]: {}\n".format(log[idx]['sys'])

        # print(prompt)

        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        except Exception as e:
            logger.error('Exception: {}'.format(e))
            # print('Exception: {}'.format(e))
            dial_gen['gpt-3.5-turbo_modify'] = ""
        else:
            response = completion.choices[0].message.content
            response = response.lower()

            dial_gen['gpt-3.5-turbo_modify'] = response

    def get_goal_list(self):
        goal_list = {'train': [], 'valid': [], 'test': []}
        data_types = ['train', 'valid', 'test']
        for data_type in data_types:
            data_path = os.path.join(self.data_dir, "{}.jsonl".format(data_type))
            data = load_json_by_line(data_path)
            for idx, line in enumerate(data):
                if idx + 1 == len(data) or line['conv_idx'] != data[idx + 1]['conv_idx']:
                    dialog_id = data_type + "_" + str(line['conv_idx'])
                    goal = "[emotion] " + line['emotion'] + " [situation] " + line['situation']
                    goal_list[data_type].append({'dialog_id': dialog_id, 'goal': goal})

        return goal_list

    def flatten_dial_history(self, dial_history, len_postifx, max_length):
        ctx_len = sum([len(c) for c in dial_history])

        # consider eos_token
        spare_len = max_length - len_postifx - 1
        while ctx_len >= spare_len:
            ctx_len -= len(dial_history[0])
            dial_history.pop(0)

        context = list(chain(*dial_history))
        return context

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long)

    def encode_text(self, text, tokenizer, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text
        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]
            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]
            tokens = tokens + eos_token

        encoded_text = tokenizer.encode(" ".join(tokens))
        # except eos token
        if encoded_text[-1] == tokenizer.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def generate_single_dialog_t5(self, user_goal):
        self.simulator_model.to(device)  # <MOD>
        self.dialog_model.to(device2)  # <MOD>

        self.simulator_model.eval()  # <MOD>
        self.dialog_model.eval()  # <MOD>

        dial_gen = user_goal
        log = []
        dialog_history = []
        goal_state_span = user_goal['goal']
        user_utterance = None
        utterance_count = 0
        single_turn = {}

        def is_continue(dial_gen):
            if 'sys' not in single_turn and 'user' in single_turn:
                # end up with system resp
                return True
            if len(log) >= self.cfg.max_turn_num:
                # 超过固定轮数终止
                dial_gen['terminate_reason'] = '超过{}轮终止'.format(self.cfg.max_turn_num)
                return False
            # 不满足退出条件则继续循环
            return True

        while is_continue(dial_gen):
            if utterance_count & 1:  # utterance_count为奇数，ds模型生成话语阶段
                '''
                system agent:
                input: dialog history + user;
                output: response;
                '''
                utterance_count += 1

                if user_utterance is None:
                    raise Exception('Should generate user utterance first!')

                user_utterance_ids = self.encode_text(user_utterance, self.dialog_tokenizer)
                encoded_dialog_history = [self.encode_text(text, self.dialog_tokenizer) for text in dialog_history]
                context = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids),
                                                    self.dialog_tokenizer.model_max_length)
                input_ids = self.tensorize([context + user_utterance_ids + [self.dialog_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device2)  # <MOD> device to device2 ds_device

                dialog_generate = self.dialog_model.generate.__wrapped__
                torch.set_grad_enabled(False)
                model_output = dialog_generate(
                    self.dialog_model,
                    input_ids=input_ids,
                    # decoder_input_ids=bspn_decoder_input_ids,
                    eos_token_id=self.dialog_tokenizer.eos_token_id,
                    max_length=100,
                    # max_length=80,
                    # output_scores=output_scores,
                    # return_dict_in_generate=return_dict_in_generate,
                )
                torch.set_grad_enabled(True)

                system_resp = split_system_resp(self.dialog_tokenizer, model_output.cpu().numpy().tolist()[0])
                # system_resp, sys_resp_prob = split_system_resp(self.dialog_tokenizer, model_output.cpu().numpy().tolist()[0])

                system_resp = self.dialog_tokenizer.decode(system_resp, clean_up_tokenization_spaces=False).split()

                single_turn['sys'] = ' '.join(system_resp[1:-1])

                # 更新历史对话信息
                dialog_history.append(user_utterance)
                dialog_history.append(system_resp)

                log.append(single_turn.copy())
                single_turn = {}

                user_utterance = None

            else:  # utterance_count为偶数，us模型生成话语阶段
                '''
                user agent:
                input: dialog history + goal state span;
                output: user utterance;
                '''
                utterance_count += 1

                goal_state_ids = self.encode_text(goal_state_span, self.simulator_tokenizer,
                                                  bos_token=definitions.BOS_GOAL_TOEKN,
                                                  eos_token=definitions.EOS_GOAL_TOKEN)  # 编码goal state
                encoded_dialog_history = [self.encode_text(text, self.simulator_tokenizer) for text in dialog_history]  # 编码对话历史
                context = self.flatten_dial_history(encoded_dialog_history, len(goal_state_ids),
                                                    self.simulator_tokenizer.model_max_length)  # flatten并截断对话历史
                input_ids = self.tensorize([context + goal_state_ids + [
                    self.simulator_tokenizer.eos_token_id]])  # input: dialog history + goal state span;
                input_ids = input_ids.to(device)

                generate_with_graph = self.simulator_model.generate.__wrapped__
                torch.set_grad_enabled(False)  # torch.set_grad_enabled(if_usr_need_grad)
                model_output = generate_with_graph(
                    self.simulator_model,
                    input_ids=input_ids,
                    eos_token_id=self.simulator_tokenizer.eos_token_id,
                    # max_length=200,
                    max_length=100,
                    # output_scores=output_scores,
                    # return_dict_in_generate=return_dict_in_generate,
                )
                torch.set_grad_enabled(True)

                user_utterance = split_user_resp(self.simulator_tokenizer, model_output.cpu().numpy().tolist()[0])
                # user_utterance = split_user_resp(self.simulator_tokenizer, user_utterance_output[0])
                # print("encode_user_utterance: ", user_utterance)

                user_utterance = self.simulator_tokenizer.decode(user_utterance,
                                                                 clean_up_tokenization_spaces=False).split(' ')
                # print("user_utterance: ", user_utterance)

                single_turn['user'] = ' '.join(user_utterance[1:-1])
                # print("single_turn_user: ", single_turn['user'])

        dial_gen['log'] = log
        return dial_gen

    def generate_single_dialog_t5blender(self, user_goal):
        self.simulator_model.to(device)  # <MOD>
        self.dialog_model.to(device2)  # <MOD>

        dial_gen = user_goal
        log = []
        dialog_history = []
        goal_state_span = user_goal['goal']
        user_utterance = None
        utterance_count = 0
        single_turn = {}

        def is_continue(dial_gen):
            if 'sys' not in single_turn and 'user' in single_turn:
                # end up with system resp
                return True
            if len(log) >= self.cfg.max_turn_num:
                # 超过固定轮数终止
                dial_gen['terminate_reason'] = '超过{}轮终止'.format(self.cfg.max_turn_num)
                return False
            # 不满足退出条件则继续循环
            return True

        while is_continue(dial_gen):
            if utterance_count & 1:  # utterance_count为奇数，ds模型生成话语阶段
                '''
                system agent:
                input: dialog history + user;
                output: response;
                '''
                utterance_count += 1

                if user_utterance is None:
                    raise Exception('Should generate user utterance first!')

                new_user_utterance = " ".join(user_utterance[1:-1])
                new_dialog_history = [" ".join(text[1:-1]) for text in dialog_history]
                inputs = ""
                for idx, text in enumerate(new_dialog_history):
                    if idx != 0:
                        inputs += " </s> <s> "
                    inputs += text
                if inputs != "":
                    inputs = inputs + " </s> <s> " + new_user_utterance
                else:
                    inputs = new_user_utterance
                # print(inputs)

                input_ids = self.dialog_tokenizer([inputs], return_tensors="pt")
                input_ids = input_ids.to(device2)

                system_resp_ids = self.dialog_model.generate(**input_ids)
                system_resp = self.dialog_tokenizer.batch_decode(system_resp_ids, skip_special_tokens=True)[0]

                single_turn['sys'] = system_resp

                # 更新历史对话信息
                dialog_history.append(user_utterance)
                dialog_history.append(['<bos_resp>'] + system_resp.split() + ['<eos_resp>'])

                log.append(single_turn.copy())
                single_turn = {}

                user_utterance = None

            else:  # utterance_count为偶数，us模型生成话语阶段
                '''
                user agent:
                input: dialog history + goal state span;
                output: user utterance;
                '''
                utterance_count += 1

                goal_state_ids = self.encode_text(goal_state_span, self.simulator_tokenizer,
                                                  bos_token=definitions.BOS_GOAL_TOEKN,
                                                  eos_token=definitions.EOS_GOAL_TOKEN)  # 编码goal state
                encoded_dialog_history = [self.encode_text(text, self.simulator_tokenizer) for text in dialog_history]  # 编码对话历史
                context = self.flatten_dial_history(encoded_dialog_history, len(goal_state_ids),
                                                    self.simulator_tokenizer.model_max_length)  # flatten并截断对话历史
                input_ids = self.tensorize([context + goal_state_ids + [
                    self.simulator_tokenizer.eos_token_id]])  # input: dialog history + goal state span;
                input_ids = input_ids.to(device)

                generate_with_graph = self.simulator_model.generate.__wrapped__
                torch.set_grad_enabled(False)  # torch.set_grad_enabled(if_usr_need_grad)
                model_output = generate_with_graph(
                    self.simulator_model,
                    input_ids=input_ids,
                    eos_token_id=self.simulator_tokenizer.eos_token_id,
                    # max_length=200,
                    max_length=100,
                    # output_scores=output_scores,
                    # return_dict_in_generate=return_dict_in_generate,
                )
                # torch.set_grad_enabled(True)

                user_utterance = split_user_resp(self.simulator_tokenizer, model_output.cpu().numpy().tolist()[0])
                # user_utterance = split_user_resp(self.simulator_tokenizer, user_utterance_output[0])
                # print("encode_user_utterance: ", user_utterance)

                user_utterance = self.simulator_tokenizer.decode(user_utterance,
                                                                 clean_up_tokenization_spaces=False).split(' ')
                # print("user_utterance: ", user_utterance)

                single_turn['user'] = ' '.join(user_utterance[1:-1])
                # print("single_turn_user: ", single_turn['user'])

        dial_gen['log'] = log
        return dial_gen


def convert_data_format(file_path, save_path):
    '''
    conv_idx, utter_idx, utterance, emotion, situation
    '''
    # file_name = 'test'
    #
    # file_path = f'../EUS_data/{file_name}_prompt_gpt3_5_brief_continue.jsonl'
    data = load_json(file_path)

    new_modify_data = []
    for dial in data:
        dial_list = []

        delimiters = "- [modified speaker]:", "- [modified listener]:"  # "[whether modified]", "[reason]", "[modified conversation]", "- [modified speaker]:", "- [modified listener]:"
        regexPattern = '|'.join(map(re.escape, delimiters))
        modify_dialogue = dial['gpt-3.5-turbo_modify']
        modify_dialogue = re.split(regexPattern, modify_dialogue)
        modify_dialogue = [i.strip() for i in modify_dialogue]
        modify_dialogue = modify_dialogue[1:]
        for utterance in modify_dialogue:
            if utterance == '':
                continue
            dial_list.append(utterance)

        new_modify_data.append(dial_list)

    examples = []
    for dial_list, dial_info in zip(new_modify_data, data):
        example = {}

        for utter_idx, utterance in enumerate(dial_list):
            example['conv_idx'] = dial_info['dialog_id']
            example['utter_idx'] = utter_idx
            example['utterance'] = utterance
            example['emotion'] = dial_info['goal'].split()[1]  # dial_info['emotion']
            example['situation'] = " ".join(dial_info['goal'].split()[3:])  # dial_info['situation']
            examples.append(copy.deepcopy(example))

    # save_path = f'../EUS_data/version3.0/{file_name}.jsonl'
    with open(save_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    return examples


def generation_modify_train():
    # 获取两模型交互参数
    inter_cfg = inter_get_config()
    # 创建交互环境
    interaction = InteractionEnvironment(inter_cfg)

    us_bestbleu4, us_bestloss = 0, 1e8
    ds_bestbleu4, ds_bestloss = 0, 1e8

    for epoch in range(10):  # range(10):
        # 生成对话
        if os.path.exists(os.path.join(inter_cfg.generate_results_path, "tmp_output_%03d.json" % epoch)):
            pass
        else:
            logger.info("Generation Dialogue...")
            dialogs_gen = []
            random_numbers = [random.randint(0, 19530) for _ in range(200)]  # 随机选5个场景
            for idx in tqdm(random_numbers, desc="Generation:"):
                goal = interaction.goal_list[inter_cfg.interaction_type][idx]
                # print(goal)
                dial_gen = interaction.generate_single_dialog_t5(goal)

                # gpt 生成 modify dialogue
                interaction.OpenAI_API_modify(dial_gen)

                dialogs_gen.append(dial_gen)

            save_json(dialogs_gen, os.path.join(inter_cfg.generate_results_path, "tmp_output_%03d.json" % epoch))

            # 将生成的对话 转换格式为 训练数据
            convert_data_format(os.path.join(inter_cfg.generate_results_path, "tmp_output_%03d.json" % epoch),
                                "interact_output/t5_large_generate_modify/train/train_%03d.json" % epoch)

        torch.cuda.empty_cache()

        # 训练用户模拟器
        # -data_version 3.0 -agent_type us -run_type train -backbone t5-large -model_dir simulator_t5_large_data3.0_interact -epoch 2
        logger.info("User Simulator Training...")

        # interaction.simulator_model.to(torch.device("cuda:0"))
        # interaction.dialog_model.to(torch.device("cuda:1"))

        cfg = get_config()
        cfg.data_version = '3.0'
        cfg.agent_type = 'us'
        cfg.run_type = 'train'
        cfg.backbone = 't5-large'
        # cfg.aug_cutoff_ratio = 0.03
        cfg.learning_rate = 5e-4
        cfg.batch_size = 2
        cfg.epochs = 1
        cfg.model_dir = "interact_model/simulator_{}_data{}_lr{}_bs{}_interact".format(cfg.backbone, cfg.data_version, cfg.learning_rate, cfg.batch_size)
        # setattr(cfg, "model_dir", "interact_model/simulator_t5_large_data3.0_interact")

        num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

        setattr(cfg, "device", device)  # setattr(cfg, "device", torch.device("cuda:0"))
        setattr(cfg, "num_gpus", num_gpus)

        setattr(cfg, "bestbleu4", us_bestbleu4)
        setattr(cfg, "bestloss", us_bestloss)

        reader = EDReader(interaction.simulator_tokenizer, epoch)  # EDReader(tokenizer)

        runner = EDRunner(cfg)  # 初始化：iterator, reader, model
        runner.reader = reader
        runner.iterator = EDIterator(reader)
        runner.model = interaction.simulator_model
        runner.model.to(cfg.device)

        runner.train()

        us_bestbleu4 = runner.cfg.bestbleu4
        us_bestloss = runner.cfg.bestloss

        torch.cuda.empty_cache()

        # 训练对话系统
        # -data_version 3.0 -agent_type ds -run_type train -backbone t5-large -model_dir dialogue_t5_large_data3.0_interact -epoch 2
        logger.info("Dialogue System Training...")

        # interaction.simulator_model.to(torch.device("cuda:1"))
        # interaction.dialog_model.to(torch.device("cuda:0"))

        cfg = get_config()
        cfg.data_version = '3.0'
        cfg.agent_type = 'ds'
        cfg.run_type = 'train'
        cfg.backbone = 't5-large'
        # cfg.aug_cutoff_ratio = 0.03
        cfg.learning_rate = 5e-4
        cfg.batch_size = 2
        cfg.epochs = 1
        cfg.model_dir = "interact_model/dialogue_{}_data{}_lr{}_bs{}_interact".format(cfg.backbone,
                                                                                      cfg.data_version,
                                                                                      cfg.learning_rate,
                                                                                      cfg.batch_size)

        num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

        setattr(cfg, "device", device2)  # setattr(cfg, "device", torch.device("cuda:0"))
        setattr(cfg, "num_gpus", num_gpus)

        setattr(cfg, "bestbleu4", ds_bestbleu4)
        setattr(cfg, "bestloss", ds_bestloss)

        reader = EDReader(interaction.dialog_tokenizer, epoch)  # EDReader(tokenizer)

        runner = EDRunner(cfg)  # 初始化：iterator, reader, model
        runner.reader = reader
        runner.iterator = EDIterator(reader)
        runner.model = interaction.dialog_model
        runner.model.to(cfg.device)

        runner.train()

        ds_bestbleu4 = runner.cfg.bestbleu4
        ds_bestloss = runner.cfg.bestloss

        torch.cuda.empty_cache()



if __name__ == '__main__':
    export_api_key()
    client = OpenAI()

    generation_modify_train()

    # # 获取参数
    # inter_cfg = inter_get_config()
    #
    # interaction = InteractionEnvironment(inter_cfg)
    #
    # dialogs_gen = []
    # # 生成对话
    # for goal in tqdm(interaction.goal_list[cfg.interaction_type][:10], desc="Generation"):
    #     # print(goal)
    #     dial_gen = interaction.generate_single_dialog_t5blender(goal)
    #     dialogs_gen.append(dial_gen)
    #
    # save_json(dialogs_gen, cfg.generate_results_path)

    #
    # data = load_json('./interact_output/tmp_output_2.json')
    # for dial_gen in tqdm(data, desc="Modify"):
    #     interaction.OpenAI_API_modify(dial_gen)
    # save_json(data, './interact_output/tmp_output_2.json')

    # convert_data_format('./interact_output/t5_large_generate_modify/tmp_output_2.json', './interact_output/t5_large_generate_modify/train_001.json')

    # # -data_version 3.0 -agent_type us -run_type train -backbone t5-large -model_dir simulator_t5_large_data3.0_interact -epoch 2
    # cfg = get_config()
    # num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)
    # setattr(cfg, "device", device)
    # setattr(cfg, "num_gpus", num_gpus)
    #
    # reader = EDReader(interaction.simulator_tokenizer)  # EDReader(tokenizer)
    #
    # runner = EDRunner(cfg)  # 初始化：iterator, reader, model
    # runner.reader = reader
    # runner.iterator = EDIterator(reader)
    # runner.model = interaction.simulator_model
    # runner.model.to(cfg.device)
    #
    # runner.train()






