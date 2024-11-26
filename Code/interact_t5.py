import os
import torch
import random
import argparse
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer, BertForNextSentencePrediction, BertTokenizer
from utils.utils import load_json_by_line, load_json, save_json, convert_goal_dict_to_span, convert_generate_action_span_to_dict, \
update_goal_states_during_gen, get_or_create_logger, split_user_act_and_resp, split_user_resp, split_system_resp, export_api_key
from utils import definitions
from external_knowledges import MultiWozDB
from evaluator import MultiWozEvaluator, convert_results_format
from reader import MultiWOZReader
from config import get_config

# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
# )  # for exponential backoff

from openai import OpenAI

logger = get_or_create_logger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # user
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # ds

def rl_get_config():
    parser = argparse.ArgumentParser(description='RL config')
    parser.add_argument("-rl_dial_one_epoch", type=int, default=50)  # 200
    parser.add_argument("-rl_batch_size", type=int, default=1)  # 1
    parser.add_argument("-epochs", type=int, default=15)  # 20
    parser.add_argument("-simulator_path", type=str, default='./simulator/simulator_t5_large_data3.0/ckpt-epoch10')  # ./simulator_t5_small/ckpt-epoch10  ./simulator_t5_large_lr5e-4_bs4/ckpt-epoch5
    parser.add_argument("-dialog_sys_path", type=str, default='./dialogue/dialogue_t5_large_data3.0/ckpt-epoch8')  # ./dialogue_t5_small/ckpt-epoch8  ./dialogue_t5_large_lr5e-4_bs2/ckpt-epoch5
    parser.add_argument("-simulator_save_path", type=str, default=None)
    parser.add_argument("-dialog_save_path", type=str, default=None)
    parser.add_argument("-max_turn_num", type=int, default=5)
    parser.add_argument("-data_dir", type=str, default='./data/empathetic_dialogues')
    # parser.add_argument("-model_dir", type=str, default="dialogue_t5_small")
    parser.add_argument("-discount_factor", type=float, default=0.99)
    parser.add_argument('-rl_lr', type=float, default=1e-5, help='learning rate for reinforcement learning')  # 0.0001
    parser.add_argument('-grad_clip', type=float, default=1)
    parser.add_argument("-seed", type=int, default=1998)
    parser.add_argument('-gpt_model', type=str, default='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4'])
    parser.add_argument('-do_rl_training', action="store_true")
    # parser.add_argument('-use_ppl_as_reward', action="store_true")
    # parser.add_argument('-ppl_ckpt', type=str, default='./gpt_lm_model_lr_1e_4_sentence/ckpt-epoch6')
    # parser.add_argument('-use_nsp_score_as_reward', action="store_true")
    # parser.add_argument('-nsp_ckpt', type=str, default='./bert_nsp_model_lr_1e_5_1/ckpt-epoch9')
    # parser.add_argument('-gpt_score_ckpt', type=str, default='./bart_score_gpt_lm_model_lr_1e_4/ckpt-epoch6')
    # parser.add_argument('-nsp_coef', type=float, default=0.1)
    # parser.add_argument('-ppl_coef', type=float, default=0.1)
    # parser.add_argument('-use_bart_score', action="store_true")
    # parser.add_argument('-use_gpt_score_as_reward', action="store_true")
    # parser.add_argument('-gpt_score_coef', type=float, default=0.1)
    parser.add_argument('-use_mean_rl_loss', action="store_true")
    parser.add_argument('-generate_results_path', type=str, default='interact_output/t5_large_us10_ds8_generate_results.json')  # output.json
    parser.add_argument('-interaction_type', type=str, default='test', choices=['test', 'dev'])
    parser.add_argument('-model_name', type=str, default='t5-large', choices=['t5-small', 't5-large'])
    parser.add_argument('-data_version', type=str, default='3.0', choices=['1.0', '2.0', '3.0'], help="1.0: 原始版本 2.0: 续写版本")
    args = parser.parse_args()

    return args


class InteractionEnvironment(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.simulator_model = self.load_simulator(self.cfg.simulator_path)
        self.dialog_model = self.load_system(self.cfg.dialog_sys_path)
        self.simulator_tokenizer = self.load_simulator_tokenizer(self.cfg.simulator_path)
        self.dialog_tokenizer = self.load_sys_tokenizer(self.cfg.dialog_sys_path)
        self.data_dir = self.cfg.data_dir
        self.goal_list = self.get_goal_list()
        self.train_data = load_json_by_line('./data/empathetic_dialogues/data_3.0/train.jsonl')  # <MOD>
        self.client = OpenAI()  # <MOD>
        self.good_count = 0
        self.bad_count = 0
        self.okay_count = 0
        self.error_count = 0
        self.not_find_count = 0


    def load_simulator(self, model_path):
        logger.info("Load simulator model from {}".format(model_path))
        if not os.path.exists(model_path):
            raise Exception('Model path is invalid!')
        return T5ForConditionalGeneration.from_pretrained(model_path)

    def load_system(self, model_path):
        logger.info("Load system model from {}".format(model_path))
        if not os.path.exists(model_path):
            raise Exception('Model path is invalid!')
        return T5ForConditionalGeneration.from_pretrained(model_path)

    def load_simulator_tokenizer(self, tokenizer_path):
        logger.info("Load simulator tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('Tokenizer path is invalid!')
        return T5Tokenizer.from_pretrained(tokenizer_path)

    def load_sys_tokenizer(self, tokenizer_path):
        logger.info("Load tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('Tokenizer path is invalid!')
        return T5Tokenizer.from_pretrained(tokenizer_path)

    def get_goal_list(self):
        goal_list = {'train': [], 'valid': [], 'test': []}
        data_types = ['train', 'valid', 'test']
        for data_type in data_types:
            # data_path = ''  # <MOD>
            # if self.cfg.data_version == '1.0':
            #     data_path = os.path.join(self.data_dir, "{}.jsonl".format(data_type))
            # elif self.cfg.data_version == '2.0':
            #     data_path = os.path.join(self.data_dir, "{}_gpt3_5.jsonl".format(data_type))
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

    def generate_single_dialog(self, user_goal, with_logprob=False, agent=None):
        t1, t2, t3 = 0, 0, 0
        self.simulator_model.to(device)  # <MOD>
        self.dialog_model.to(device2)  # <MOD>

        dial_gen = {user_goal['dialog_id']: {'goal': user_goal['goal']}}
        log = []
        dialog_history = []
        goal_state_span = user_goal['goal']
        user_utterance = None
        utterance_count = 0
        single_turn = {}

        if with_logprob:
            output_scores = True
            return_dict_in_generate = True
        else:
            output_scores = False
            return_dict_in_generate = False

        if_sys_need_grad = True if agent is not None and agent == 'sys' else False
        if_usr_need_grad = True if agent is not None and agent == 'usr' else False

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

                # time1 = time.time()

                user_utterance_ids = self.encode_text(user_utterance, self.dialog_tokenizer)
                encoded_dialog_history = [self.encode_text(text, self.dialog_tokenizer) for text in dialog_history]
                context = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids),
                                                    self.dialog_tokenizer.model_max_length)
                input_ids = self.tensorize([context + user_utterance_ids + [self.dialog_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device2)  # <MOD> device to device2 ds_device

                # time2 = time.time()
                # t1 += time2 - time1

                dialog_generate = self.dialog_model.generate.__wrapped__
                torch.set_grad_enabled(if_sys_need_grad)
                model_output = dialog_generate(
                    self.dialog_model,
                    input_ids=input_ids,
                    # decoder_input_ids=bspn_decoder_input_ids,
                    eos_token_id=self.dialog_tokenizer.eos_token_id,
                    max_length=100,
                    # max_length=80,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                )
                torch.set_grad_enabled(True)

                # time3 = time.time()
                # t2 += time3 - time2  # 这一块耗时最多

                if with_logprob:
                    resp_outputs = model_output.sequences.cpu().numpy().tolist()
                    resp_prob = torch.max(torch.stack(model_output.scores, dim=1).softmax(-1), dim=-1).values[0]
                else:
                    resp_outputs = model_output.cpu().numpy().tolist()
                    resp_prob = None

                system_resp = split_system_resp(self.dialog_tokenizer, resp_outputs[0])
                # system_resp, sys_resp_prob = split_system_resp(self.dialog_tokenizer, model_output.cpu().numpy().tolist()[0])

                if with_logprob:
                    single_turn['sys_prob'] = resp_prob

                system_resp = self.dialog_tokenizer.decode(system_resp, clean_up_tokenization_spaces=False).split()

                # time4 = time.time()
                # t3 += time4 - time3

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
                encoded_dialog_history = [self.encode_text(text, self.simulator_tokenizer) for text in
                                          dialog_history]  # 编码对话历史
                context = self.flatten_dial_history(encoded_dialog_history, len(goal_state_ids),
                                                    self.simulator_tokenizer.model_max_length)  # flatten并截断对话历史

                input_ids = self.tensorize([context + goal_state_ids + [
                    self.simulator_tokenizer.eos_token_id]])  # input: dialog history + goal state span;

                input_ids = input_ids.to(device)

                generate_with_graph = self.simulator_model.generate.__wrapped__
                torch.set_grad_enabled(if_usr_need_grad)
                model_output = generate_with_graph(
                    self.simulator_model,
                    input_ids=input_ids,
                    eos_token_id=self.simulator_tokenizer.eos_token_id,
                    # max_length=200,
                    max_length=100,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                )
                torch.set_grad_enabled(True)

                if with_logprob:
                    user_utterance_output = model_output.sequences.cpu().numpy().tolist()
                    user_utterance_prob = torch.max(torch.stack(model_output.scores, dim=1).softmax(-1), dim=-1).values[0]
                else:
                    user_utterance_output = model_output.cpu().numpy().tolist()
                    user_utterance_prob = None

                # user_utterance = split_user_resp(self.simulator_tokenizer, model_output.cpu().numpy().tolist()[0])
                user_utterance = split_user_resp(self.simulator_tokenizer, user_utterance_output[0])
                # print("encode_user_utterance: ", user_utterance)

                if with_logprob:
                    single_turn['user_prob'] = user_utterance_prob

                user_utterance = self.simulator_tokenizer.decode(user_utterance,
                                                                 clean_up_tokenization_spaces=False).split(' ')
                # print("user_utterance: ", user_utterance)

                single_turn['user'] = ' '.join(user_utterance[1:-1])
                # print("single_turn_user: ", single_turn['user'])

        dial_gen['log'] = log
        return dial_gen

    def update_model(self, loss, agent):
        '''
        agent: sys or usr
        '''
        assert agent in ['sys', 'usr']
        loss.backward()
        if agent == 'sys':
            torch.nn.utils.clip_grad_norm_(self.dialog_model.parameters(), self.cfg.grad_clip)
            self.rl_sys_optimizer.step()
            self.rl_sys_optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(self.simulator_model.parameters(), self.cfg.grad_clip)
            self.rl_usr_optimizer.step()
            self.rl_usr_optimizer.zero_grad()

    def get_rl_loss(self, gen_dial_batch, agent):
        '''
        agent: sys or usr
        '''
        assert agent in ['sys', 'usr']
        rl_loss = 0
        turn_num = 0

        for dial_id in gen_dial_batch:
            gen_dial = gen_dial_batch[dial_id]
            for turn in gen_dial:
                turn_rl_loss = 0
                if agent == 'sys':
                    prob = turn['sys_prob']  # torch.cat((turn['bspn_prob'], turn['sys_act_resp_prob']))
                    assert prob.shape[0] == len(turn['sys_rewards'])

                    # reward = turn['sys_rewards'][-1]
                    # assert reward == 1 or reward == 0 or reward == 0.5
                    # if reward == 0.5:
                    #     reward = 0

                    for i in range(len(prob)):
                        # turn_rl_loss += -1 * torch.log(prob[i]) * turn['sys_rewards'][i]  # (+inf, 0) * (0, 1)
                        # 修改后的 rl_loss:  # <MOD>
                        # if reward == 1:
                        #     turn_rl_loss += torch.log(1 - prob[i])
                        # else:
                        #     turn_rl_loss += -1 * torch.log(1 - prob[i])
                        turn_rl_loss += torch.log(1 - prob[i]) * turn['usr_rewards'][i]
                    if self.cfg.use_mean_rl_loss:
                        turn_rl_loss /= len(prob)
                elif agent == 'usr':
                    prob = turn['user_prob'] # turn['user_act_resp_prob']
                    assert prob.shape[0] == len(turn['usr_rewards'])

                    reward = turn['usr_rewards'][-1]
                    assert reward == 1 or reward == 0 or reward == 0.5
                    if reward == 0.5:
                        reward = 0

                    for i in range(len(prob)):
                        # turn_rl_loss += -1 * torch.log(prob[i]) * turn['usr_rewards'][i]
                        # 修改后的 rl_loss:  # <MOD>
                        # if reward == 1:
                        #     turn_rl_loss += torch.log(1 - prob[i])
                        # else:
                        #     turn_rl_loss += -1 * torch.log(1 - prob[i])
                        turn_rl_loss += torch.log(1 - prob[i]) * turn['usr_rewards'][i]
                    if self.cfg.use_mean_rl_loss:
                        turn_rl_loss /= len(prob)
                rl_loss += turn_rl_loss
                turn_num += 1

        return rl_loss / turn_num

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    # def completion_with_backoff(**kwargs):
    #     return openai.ChatCompletion.create(**kwargs)

    def OpenAI_API_eval(self, dial_id, log):
        '''
        未实现
        '''
        emotion = log[0]['emotion']
        situation = log[0]['situation']

        origin_dial = ""
        tmp_count = 0
        conv_idx = (int)(dial_id.split('_')[1])
        for uttr in self.train_data:
            if uttr['conv_idx'] == conv_idx:
                if tmp_count & 1:
                    origin_dial += ('Listener: ' + uttr['utterance'] + '\n')
                else:
                    origin_dial += ('Speaker: ' + uttr['utterance'] + '\n')
                tmp_count += 1

        generate_dial = f'''Speaker: {log[0]['user']}
Listener: {log[0]['resp_gen']}
Speaker: {log[1]['user']}
Listener: {log[1]['resp_gen']}
Speaker: {log[2]['user']}
Listener: {log[2]['resp_gen']}
Speaker: {log[3]['user']}
Listener: {log[3]['resp_gen']}
Speaker: {log[4]['user']}
Listener: {log[4]['resp_gen']}
'''
        # random_num = random.random()
        # if random_num < 0.5:
        #     first_dialogue = origin_dial
        #     second_dialogue = generate_dial
        # else:
        #     first_dialogue = generate_dial
        #     second_dialogue = origin_dial

        prompt = f'''[Task Description] I will give you two dialogues. The two sides of the dialogues are Listener and Speaker, respectively.\
Speaker feel {emotion} because {situation}. \
Speaker shared these emotions with Listener in dialog, expecting empathy and understanding from them. \
The two dialogues are as follows:

[First Dialogue]
{origin_dial}

[Second Dialogue]
{generate_dial}

[Question]
In such open-ended dialogs, good listeners demonstrate coherence and maintain a good conversation flow, they display a \
likeable personality and understanding ofthe speaker. On the contrary, bad listeners don't follow the context and don't \
show much interest in the conversation. Please choose the one you think is better from the two dialogues, \
choosing from [First Dialogue] and [Second Dialogue] options.
'''
        # print(prompt)
        # exit()

#         prompt = f'''I am a Speaker, feeling {emotion} because {situation}. I shared these emotions with a Listener in a dialog, expecting empathy and understanding from them. Our dialog went as follows.
# Speaker: {user1}
# Listener: {sys1}
# Speaker: {user2}
# Listener: {sys2}
# In such open-ended dialogs, good listeners demonstrate coherence and maintain a good conversation flow, they display a likeable personality and understanding ofthe speaker. On the contrary, bad listeners don't follow the context and don't show much interest in the conversation. I would rate the Listener in my dialog as ___, choosing from "Bad", "Okay", and "Good" options.'''
#         # print(prompt)

        try:
            completion = self.client.chat.completions.create(
                model=self.cfg.gpt_model, # "gpt-4",
                # messages=[
                #   {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                #   {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
                # ]
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        except Exception as e:
            logger.info('Exception: {}'.format(e))
            self.error_count += 1
            return 0.5

        response = completion.choices[0].message.content
        # print(response)
        # score = None
        if response.find("[First Dialogue]") != -1 or response.find("first dialogue") != -1:
            score = 0
            self.bad_count += 1
            # if random_num < 0.5:
            #     score = 0
            #     self.bad_count += 1
            # else:
            #     score = 1
            #     self.good_count += 1
        elif response.find("[Second Dialogue]") != -1 or response.find("second dialogue") != -1:
            score = 1
            self.good_count += 1
            # if random_num < 0.5:
            #     score = 1
            #     self.good_count += 1
            # else:
            #     score = 0
            #     self.bad_count += 1
        else:
            score = 0.5
            self.not_find_count += 1

        # print(score)
        # exit()
        return score


        # # print(completion.choices[0].message.content)
        # response = completion.choices[0].message.content
        #
        # score = None
        # if response.find('''"Good"''') != -1:
        #     score = 1
        #     self.good_count += 1
        # elif response.find('''"Okay"''') != -1:
        #     score = 0.5
        #     self.okay_count += 1
        # elif response.find('''"Bad"''') != -1:
        #     score = 0
        #     self.bad_count += 1
        # else:
        #     score = 0.5
        #     self.not_find_count += 1
        #
        # # print(score)
        # # exit()
        #
        # # score = random.randint(0, 1)
        # return score


    def get_success_reward(self, gen_dial_batch):
        '''
        assgining user rewards to turn['usr_rewards']
        assgining system rewards to turn['sys_rewards']
        '''
        batch_rewards = []
        for dial_id in gen_dial_batch:
            # API测评
            score = self.OpenAI_API_eval(dial_id, gen_dial_batch[dial_id])  # success, _ = evaluator.e2e_eval({dial_id: gen_dial_batch[dial_id]}, online_eval=True)
            # 根据评分设置reward
            reward = score  # reward = 1 - score  # reward = score  # <MOD>

            # If a dialog is successful, we set the reward of each turn to 1
            self.all_rewards.append(reward)  # 这句好像没用
            batch_rewards.append(reward)

            for turn in gen_dial_batch[dial_id]:
                usr_r, sys_r = reward, reward

                usr_rewards = []
                sys_rewards = []

                usr_len = len(turn['user_prob'])  # len(turn['user_act_resp_prob'])
                sys_len = len(turn['sys_prob'])  # len(turn['bspn_prob']) + len(turn['sys_act_resp_prob'])

                for _ in range(usr_len):
                    usr_rewards.insert(0, usr_r)
                    usr_r = usr_r * self.cfg.discount_factor

                for _ in range(sys_len):
                    sys_rewards.insert(0, sys_r)
                    sys_r = sys_r * self.cfg.discount_factor

                turn['usr_rewards'] = usr_rewards
                turn['sys_rewards'] = sys_rewards

        return np.mean(batch_rewards)

    def rl_validation(self):
        '''
        计算强化学习后的得分，未实现
        '''
        score = 0.0
        # dialogs_gen = []
        # for goal in tqdm(self.goal_list['valid'][:10], desc='Validation'):
        #     dial_gen = interaction.generate_single_dialog(goal)
        #     dialogs_gen.append(dial_gen)
        # # success, match = evaluator.e2e_eval(dialogs_gen, online_eval=True)

        cfg = get_config()
        cfg.data_version = '3.0'
        cfg.run_type = 'predict'
        cfg.predict_agent_type = 'us'
        cfg.pred_data_type = 'test'
        cfg.ckpt = './simulator/simulator_t5_large_data3.0/simulator_rl_epoch_2'
        cfg.output = 'inference.json'
        cfg.batch_size = 32

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

        setattr(cfg, "device", device)
        setattr(cfg, "num_gpus", num_gpus)

        logger.info("Device: %s (the number of GPUs: %d)", str(device), num_gpus)

        reader = EDReader(cfg)
        iterator = EDIterator(reader)
        model = xxx

        model.eval()
        pred_batches, _, _, _ = iterator.get_batches(
            cfg.pred_data_type, cfg.batch_size,
            cfg.num_gpus, excluded_domains=self.cfg.excluded_domains, num_dialogs=-1)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            for turn_batch in iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context = iterator.flatten_dial_history(
                        dial_history[t], len(turn['user']), self.cfg.context_size
                    )

                    encoder_input_ids = context + turn['user'] + [reader.eos_token_id]
                    batch_encoder_input_ids.append(iterator.tensorize(encoder_input_ids))

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=reader.pad_token_id)
                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)
                attention_mask = torch.where(batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                with torch.no_grad():
                    model_outputs = model.generate(
                        input_ids=batch_encoder_input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=reader.eos_token_id,
                        max_length=200
                    )

                model_outputs = model_outputs.cpu().numpy().tolist()

                for t, turn in enumerate(turn_batch):
                    system_resp = split_system_resp(reader.tokenizer, model_outputs[t])
                    system_resp = reader.tokenizer.decode(system_resp,
                                                               clean_up_tokenization_spaces=False).split()
                    system_resp = ' '.join(system_resp[1:-1])
                    turn['sys_gen'] = system_resp

                    pv_text = copy.copy(turn['user'])
                    pv_text = pv_text + turn['resp']
                    dial_history[t].append(pv_text)

            result = iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        return results


        return score


    def train_RL(self):
        self.all_rewards = []  # rewards container 好像没用

        # 优化器
        self.rl_sys_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dialog_model.parameters()),
                                                 lr=self.cfg.rl_lr)
        self.rl_usr_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.simulator_model.parameters()),
                                                 lr=self.cfg.rl_lr)

        best_score = 0  # 记录最好得分，以及epoch数
        best_score_epoch = 0
        random.shuffle(self.goal_list['valid'])

        for epoch in range(1, self.cfg.epochs + 1):  # <MOD>
            self.good_count = 0
            self.okay_count = 0
            self.bad_count = 0
            self.error_count = 0
            self.not_find_count = 0
            self.cfg.rl_dial_one_epoch = min(len(self.goal_list['train']), self.cfg.rl_dial_one_epoch)
            n_batch = self.cfg.rl_dial_one_epoch // self.cfg.rl_batch_size

            random.shuffle(self.goal_list['train'])

            epoch_avg_rewards = 0
            epoch_avg_rl_loss = 0

            for agent in ['usr']:  # ['usr', 'sys']:
                for i in tqdm(range(n_batch), desc='Reinforcement Learning ({})'.format(agent)):
                    torch.cuda.empty_cache()  # <MOD>
                    try:
                        start_idx = i * self.cfg.rl_batch_size
                        end_idx = (i + 1) * self.cfg.rl_batch_size
                        dial_goals = self.goal_list['train'][start_idx:end_idx]

                        gen_dial_batch = []
                        for goal in dial_goals:
                            dial_gen = self.generate_single_dialog(goal, with_logprob=True, agent=agent)
                            gen_dial_batch.append(dial_gen)

                        # # <MOD>
                        # obj = gen_dial_batch[0]
                        # for turn in obj['log']:
                        #     for k, v in turn.items():
                        #         if torch.is_tensor(v):
                        #             turn[k] = v.tolist()
                        # save_json(gen_dial_batch, 'data/tmp/gen_dial_batch.json')
                        # # print(gen_dial_batch)

                        gen_dial_batch = convert_results_format(gen_dial_batch)

                        # # <MOD>
                        # for id, log in gen_dial_batch.items():
                        #     for turn in log:
                        #         for k, v in turn.items():
                        #             if torch.is_tensor(v):
                        #                 turn[k] = v.tolist()
                        # save_json(gen_dial_batch, 'data/tmp/processed_gen_dial_batch.json')

                        avg_rewards = self.get_success_reward(gen_dial_batch)  # self.get_success_reward(gen_dial_batch, evaluator)  # 1.reward函数怎么写得，reward怎么获得的

                        # # <MOD>
                        # # print("avg_rewards: ", avg_rewards)
                        # for id, log in gen_dial_batch.items():
                        #     for turn in log:
                        #         for k, v in turn.items():
                        #             if torch.is_tensor(v):
                        #                 turn[k] = v.tolist()
                        # save_json(gen_dial_batch, 'data/tmp/processed_gen_dial_batch_with_reward.json')
                        # exit()

                        rl_loss = self.get_rl_loss(gen_dial_batch, agent)  # 2.rl_loss是什么
                        epoch_avg_rl_loss += rl_loss.item()
                        self.update_model(rl_loss, agent)  # 3. update怎么更新的

                        del rl_loss
                        del gen_dial_batch
                        torch.cuda.empty_cache()
                        epoch_avg_rewards += avg_rewards

                    except RuntimeError as e:
                        # logger.info('CUDA Out of Memory.')  # <MOD>
                        self.error_count += 1
                        logger.info('Exception: {}'.format(e))


            logger.info('Epoch: {}; Avg rewards: {}; Avg RL Loss: {}'.format(epoch, epoch_avg_rewards / (2 * n_batch),
                                                                             epoch_avg_rl_loss / (2 * n_batch)))

            logger.info('Epoch: {}, Good_count = {}, Okay_count = {}, Bad_count = {}, Error_count = {}, Not_Find_count = {}'.format(
                epoch, self.good_count, self.okay_count, self.bad_count, self.error_count, self.not_find_count))

            # <MOD> 直接把每一轮都存下来
            if epoch % 5 == 0:
                simulator_dir = os.path.dirname(self.cfg.simulator_path)
                dialog_dir = os.path.dirname(self.cfg.dialog_sys_path)
                self.simulator_model.save_pretrained(
                    os.path.join(simulator_dir, self.cfg.simulator_save_path + '_epoch_{}'.format(epoch)))  # <MOD>
                self.simulator_tokenizer.save_pretrained(
                    os.path.join(simulator_dir, self.cfg.simulator_save_path + '_epoch_{}'.format(epoch)))  # <MOD>
            # self.dialog_model.save_pretrained(
            #     os.path.join(dialog_dir, self.cfg.dialog_save_path + '_epoch_{}'.format(epoch)))
            # self.dialog_tokenizer.save_pretrained(
            #     os.path.join(dialog_dir, self.cfg.dialog_save_path + '_epoch_{}'.format(epoch)))

            # score = self.rl_validation(evaluator_dev)  # 强化学习后验证得分
            # if score > best_score:
            #     best_score = score
            #     best_score_epoch = epoch
            #     simulator_dir = os.path.dirname(self.cfg.simulator_path)
            #     dialog_dir = os.path.dirname(self.cfg.dialog_sys_path)
            #     self.simulator_model.save_pretrained(
            #         os.path.join(simulator_dir, self.cfg.simulator_save_path + '_epoch_{}'.format(epoch)))
            #     self.simulator_tokenizer.save_pretrained(
            #         os.path.join(simulator_dir, self.cfg.simulator_save_path + '_epoch_{}'.format(epoch)))
            #     self.dialog_model.save_pretrained(
            #         os.path.join(dialog_dir, self.cfg.dialog_save_path + '_epoch_{}'.format(epoch)))
            #     self.dialog_tokenizer.save_pretrained(
            #         os.path.join(dialog_dir, self.cfg.dialog_save_path + '_epoch_{}'.format(epoch)))
            #
            # logger.info(
            #     'Epoch: {}; Score: {}; Best Score: {}; Best epoch: {}'.format(epoch, score, best_score, best_score_epoch))


if __name__ == '__main__':
    # 联网
    export_api_key(9999)

    # 获取参数
    cfg = rl_get_config()
    # if cfg.data_version == '1.0':
    #     cfg.data_dir = 'data/empathetic_dialogues/original_data'
    # elif cfg.data_version == '2.0':
    #     cfg.data_dir = 'data/empathetic_dialogues/datagpt3_5'
    cfg.data_dir = os.path.join(cfg.data_dir, "data_{}".format(cfg.data_version))
    cfg.simulator_save_path = 'simulator_rl_dc{}_lr{}_gc{}_{}_newloss'.format(cfg.discount_factor, cfg.rl_lr, cfg.grad_clip, cfg.gpt_model)
    cfg.dialog_save_path = 'dialog_rl_dc{}_lr{}_gc{}_{}_newloss'.format(cfg.discount_factor, cfg.rl_lr, cfg.grad_clip, cfg.gpt_model)

    interaction = InteractionEnvironment(cfg)

    # save_json(interaction.goal_list, 'data/empathetic_dialogues/goal_list.json')

    if cfg.do_rl_training:
        # random seeds
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

        interaction.train_RL()
    else:
        # dialogs_gen = []
        # # 生成对话
        # for goal in tqdm(interaction.goal_list[cfg.interaction_type]):
        #     dial_gen = interaction.generate_single_dialog(goal)
        #     dialogs_gen.append(dial_gen)
        #
        # save_json(dialogs_gen, cfg.generate_results_path)

        dialogs_gen = []
        # t1, t2, t3 = 0, 0, 0
        random_numbers = list(range(100))  # [random.randint(0, 2546) for _ in range(100)]
        for idx in tqdm(random_numbers, desc="Generation:"):
            goal = interaction.goal_list[cfg.interaction_type][idx]
            dial_gen = interaction.generate_single_dialog(goal)
            dialogs_gen.append(dial_gen)
        save_json(dialogs_gen, cfg.generate_results_path)
