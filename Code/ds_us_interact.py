import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from itertools import chain
from utils.utils import get_or_create_logger, load_json, save_json, split_user_resp, export_api_key
from transformers import T5ForConditionalGeneration, T5Tokenizer, BlenderbotSmallForConditionalGeneration, AutoTokenizer
from openai import OpenAI
from utils import definitions
import requests
import json


logger = get_or_create_logger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # user
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # ds
export_api_key(9999)

def get_config():
    parser = argparse.ArgumentParser(description='ds us interact and evaluation')
    parser.add_argument("-simulator_path", type=str,
                        default='./simulator/simulator_t5_large_data3.0/ckpt-epoch10')  # ./simulator_t5_small/ckpt-epoch4  ./simulator_t5_large_lr5e-4_bs4/ckpt-epoch5
    parser.add_argument("-dialog_sys_path", type=str,
                        default='./blenderbot_small-90M')  # ./dialogue_t5_small/ckpt-epoch6  ./dialogue_t5_large_lr5e-4_bs2/ckpt-epoch5
    parser.add_argument('-us_model_name', type=str, default='gpt-3.5-turbo', choices=['t5-small', 't5-large', 'gpt-3.5-turbo'])
    parser.add_argument('-ds_model_name', type=str, default='gpt-3.5-turbo', choices=['t5-small', 't5-large', 'blenderbot_small', 'gpt-3.5-turbo'])
    parser.add_argument("-max_turn_num", type=int, default=5)
    parser.add_argument("-data_path", type=str, default='./data/empathetic_dialogues/ieval_data.json')
    # parser.add_argument("-seed", type=int, default=1998)
    parser.add_argument('-generate_results_path', type=str,
                        default='interact_output/simulator_gpt_and_system_gpt/simulator_gpt-3.5-turbo_and_system_gpt-3.5-turbo.json')  # output.json
    # parser.add_argument('-interaction_type', type=str, default='test', choices=['test', 'dev'])
    args = parser.parse_args()

    return args

# 计算 gpt_eval_score 和 human_score 的 pearson相关系数
def calculate_pearson(gpt_eval_score, human_score):
    # 定义 x 和 y 的数据
    x = gpt_eval_score
    y = human_score

    # 绘制散点图
    plt.scatter(x, y)

    # 添加标题和标签
    plt.title("Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 显示图表
    plt.show()

    data = {
        'gpt_eval_score': gpt_eval_score,
        'human_score': human_score
    }
    df = pd.DataFrame(data)
    print(df)
    print(df.mean())

    pearson_corr = df.corr(method='pearson')
    return pearson_corr


def OpenAI_API_eval(dial_gen):
    log = dial_gen['log']
    goal =  dial_gen['goal'].split()
    emotion = goal[1]
    situation = ' '.join(goal[3:])

    user1 = log[0]['user']
    sys1 = log[0]['sys']

    user2 = log[1]['user']
    sys2 = log[1]['sys']

    user3 = log[2]['user']
    sys3 = log[2]['sys']

    prompt = f'''I am a Speaker, feeling {emotion} because {situation}. I shared these emotions with a Listener in a dialog, expecting empathy and understanding from them. Our dialog went as follows.
Speaker: {user1}
Listener: {sys1}
Speaker: {user2}
Listener: {sys2}
Speaker: {user3}
Listener: {sys3}
In such open-ended dialogs, good listeners demonstrate coherence and maintain a good conversation flow, they display a likeable personality and understanding ofthe speaker. On the contrary, bad listeners don't follow the context and don't show much interest in the conversation. I would rate the Listener in my dialog as ___, choosing from "Bad", "Okay", and "Good" options.'''
    # print(prompt)

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:
        # logger.info('Exception: {}'.format(e))
        print('Exception: {}'.format(e))

    response = completion.choices[0].message.content

    score = None
    if response.find('''"Good"''') != -1:
        score = 1
    elif response.find('''"Okay"''') != -1:
        score = 0.5
    elif response.find('''"Bad"''') != -1:
        score = 0
    else:
        score = -1

    dial_gen['gpt-3.5-turbo_response'] = response
    dial_gen['gpt-3.5-turbo_score'] = score


def GPT_eval_5_metric(dial_gen):
    log = dial_gen['log']

    prompt = f'''Hello, next I will give you five rounds of dialogue. The two sides of the dialogue are human and system.\
Please help me to score the system side in the dialogue on the following five dimensions. The scoring range is 1-5, \
of which 1 is the worst, and 5 is the best: Emotional empathy, Cognitive empathy, Relevence, Fluency, and Rationality, \
which respectively mean: (1) The system's ability to feel and respond with the emotions of others, following understanding, \
support, and connection in international relationships (2) The system's ability to understand and compare the ideas and \
perspectives of others, without necessarily sharing those ideas (3) The degree of relevance of the reply statement of \
the system side to the whole conversation content. (4) The fluency of the reply statements of the System side (5) \
The rationality and logical correctness of the reply statements of the System side. 
In addition, the format of your response should follow the following criteria: \
"Emotional empathy: 2, Cognitive empathy: 4, Relevance: 3, Fluency: 5, Rationality: 2". \
The numerical part of the reply should be replaced with the specific score \
you give to the dialogue replied by the system.

Human: {log[0]['user']}
System: {log[0]['sys']}
Human: {log[1]['user']}
System: {log[1]['sys']}
Human: {log[2]['user']}
System: {log[2]['sys']}
Human: {log[3]['user']}
System: {log[3]['sys']}
Human: {log[4]['user']}
System: {log[4]['sys']}
'''
    # print(prompt)

    try:
        url = "https://api.gpts.vin"  #"https://api.kuaiwenwen.top/v1/chat/completions"

        payload = json.dumps({
            "model": "gpt-3.5-turbo",  # "gpt-3.5-turbo-16k", "gpt4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 1,
            "max_tokens": 1024,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        })
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer sk-1xTAHfoeUiTfTo5227Ae008c43Fe4f44Ba1aA88d9074Eb4d',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        response = ''

    # print(response)
    dial_gen['gpt-3.5-turbo_response'] = response
    # dial_gen['gpt-3.5-turbo_Emotional_empathy_score'] = score


class InteractionEnvironment(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.simulator_model = self.load_simulator(self.cfg.simulator_path)
        self.dialog_model = self.load_system(self.cfg.dialog_sys_path)
        self.simulator_tokenizer = self.load_simulator_tokenizer(self.cfg.simulator_path)
        self.dialog_tokenizer = self.load_sys_tokenizer(self.cfg.dialog_sys_path)
        self.data_path = self.cfg.data_path
        # self.goal_list = self.get_goal_list()
        self.goal_list = load_json('./data/empathetic_dialogues/goal_list.json')[:100]

    def load_simulator(self, model_path):
        if "gpt" in self.cfg.us_model_name:
            logger.info("User simulator model is {}".format(self.cfg.us_model_name))
            return

        logger.info("Load simulator model from {}".format(model_path))
        if not os.path.exists(model_path):
            raise Exception('Simulator Model path is invalid!')
        if self.cfg.us_model_name == "t5-large" or self.cfg.us_model_name == "t5-small":
            return T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            raise Exception('US Model name is invalid!')

    def load_system(self, model_path):
        if "gpt" in self.cfg.ds_model_name:
            logger.info("System model is {}".format(self.cfg.ds_model_name))
            return

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
        if "gpt" in self.cfg.us_model_name:
            logger.info("{} User simulator don't need Tokenizer".format(self.cfg.ds_model_name))
            return

        logger.info("Load simulator tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('Simulator Tokenizer path is invalid!')
        if self.cfg.us_model_name == "t5-large" or self.cfg.us_model_name == "t5-small":
            return T5Tokenizer.from_pretrained(tokenizer_path)
        else:
            raise Exception('US Model name is invalid!')

    def load_sys_tokenizer(self, tokenizer_path):
        if "gpt" in self.cfg.ds_model_name:
            logger.info("{} System don't need Tokenizer".format(self.cfg.ds_model_name))
            return

        logger.info("Load system tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('System Tokenizer path is invalid!')
        if self.cfg.ds_model_name == "t5-large" or self.cfg.ds_model_name == "t5-small":
            return T5Tokenizer.from_pretrained(tokenizer_path)
        elif self.cfg.ds_model_name == "blenderbot_small":
            return AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            raise Exception('DS Model name is invalid!')

    def get_goal_list(self):
        goal_list = []
        data = load_json(self.cfg.data_path)

        for obj in data:
            dialog_id = obj['positive']['conv_id']
            emotion = obj['positive']['emotion']
            situation = obj['positive']['prompt']
            goal = "[emotion] " + emotion + " [situation] " + situation
            goal_list.append({'dialog_id': dialog_id, 'goal': goal})

            dialog_id = obj['negative']['conv_id']
            emotion = obj['negative']['emotion']
            situation = obj['negative']['prompt']
            goal = "[emotion] " + emotion + " [situation] " + situation
            goal_list.append({'dialog_id': dialog_id, 'goal': goal})

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

    def generate_single_dialog_us_t5_ds_blenderbot(self, user_goal):
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

                # new_user_utterance = " ".join(user_utterance[1:-1])
                # new_dialog_history = [" ".join(text[1:-1]) for text in dialog_history]
                # inputs = ""
                # for idx, text in enumerate(new_dialog_history):
                #     if idx != 0:
                #         inputs += " </s> <s> "
                #     inputs += text
                # if inputs != "":
                #     inputs = inputs + " </s> <s> " + new_user_utterance
                # else:
                #     inputs = new_user_utterance
                # # print(inputs)

                inputs = " ".join(user_utterance[1:-1])

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

    def generate_single_dialog_us_t5_ds_gpt(self, user_goal):
        self.simulator_model.to(device)  # <MOD>
        # self.dialog_model.to(device2)  # <MOD>

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
                
                ds prompt:
                You are engaging in a conversation with a human. Respond in an empathetic manner to the following using \
                on average 15 words and a maximum of 60 words.
                
                Human: xxx
                You: xxx
                Human: xxx
                You: 
                '''
                utterance_count += 1

                if user_utterance is None:
                    raise Exception('Should generate user utterance first!')

                new_user_utterance = " ".join(user_utterance[1:-1])
                new_dialog_history = [" ".join(text[1:-1]) for text in dialog_history]

                prompt = "You are engaging in a conversation with a human. Respond in an empathetic manner to \
the following using on average 15 words and a maximum of 60 words.\n\n"
                for idx, text in enumerate(new_dialog_history):
                    if idx % 2 == 0:
                        prompt += "Human: {}\n".format(text)
                    else:
                        prompt += "You: {}\n".format(text)

                prompt += "Human: {}\n".format(new_user_utterance)
                prompt += "You: "
                # print(prompt)

                try:
                    url = "https://api.kuaiwenwen.top/v1/chat/completions"

                    payload = json.dumps({
                        "model": "gpt-3.5-turbo",  # "gpt-3.5-turbo-16k", "gpt4",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 1,
                        "max_tokens": 1024,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    })
                    headers = {
                        'Accept': 'application/json',
                        'Authorization': 'Bearer sk-1xTAHfoeUiTfTo5227Ae008c43Fe4f44Ba1aA88d9074Eb4d',
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }

                    response = requests.request("POST", url, headers=headers, data=payload)
                    response = response.json()['choices'][0]['message']['content']
                except Exception as e:
                    print(e)
                    response = ''
                else:
                    pass
                finally:
                    system_resp = response

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
                encoded_dialog_history = [self.encode_text(text, self.simulator_tokenizer) for text in
                                          dialog_history]  # 编码对话历史
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

    def generate_single_dialog(self, user_goal):
        if (self.cfg.us_model_name == 't5-small' or self.cfg.us_model_name == 't5-large') and self.cfg.ds_model_name == 'blenderbot_small':
            return self.generate_single_dialog_us_t5_ds_blenderbot(user_goal)
        if (self.cfg.us_model_name == 't5-small' or self.cfg.us_model_name == 't5-large') and 'gpt' in self.cfg.ds_model_name:
            return self.generate_single_dialog_us_t5_ds_gpt(user_goal)


if __name__ == '__main__':
    # # 模型交互
    # # 获取参数
    # cfg = get_config()
    #
    # interaction = InteractionEnvironment(cfg)
    #
    # dialogs_gen = []
    # # 生成对话
    # for goal in tqdm(interaction.goal_list):
    #     # print(goal)
    #     dial_gen = interaction.generate_single_dialog(goal)
    #     dialogs_gen.append(dial_gen)
    #
    # save_json(dialogs_gen, cfg.generate_results_path)

    # gpt 打分
    # client = OpenAI()
    #
    # # cnt = 0
    # data = load_json('./interact_output/t5_large_us15_blenderbot_small_generate_results.json')
    # for dial_gen in tqdm(data, desc="Evaluation"):
    #     if dial_gen['gpt-3.5-turbo_score'] == -1:
    #         # cnt += 1
    #         OpenAI_API_eval(dial_gen)
    # # print("cnt = {}".format(cnt))
    #
    # save_json(data, './interact_output/t5_large_us15_blenderbot_small_generate_results.json')

    # GPT打5个分数
    # data = load_json('./interact_output/simulator_t5_large_data3.0_and_system_blenderbot/simulator_t5_large_data3.0_and_system_blenserbot_small.json')
    # data = data[:10]
    # for dial_gen in tqdm(data, desc="Evaluation"):
    #     GPT_eval_5_metric(dial_gen)
    # save_json(data, './interact_output/simulator_t5_large_data3.0_and_system_blenderbot/simulator_t5_large_data3.0_and_system_blenserbot_small_withscore.json')

    data = load_json('./interact_output/simulator_t5_large_data3.0_and_system_gpt/simulator_t5_large_data3.0_and_system_gpt-3.5-turbo.json')
    data = data
    for dial_gen in tqdm(data, desc="Evaluation"):
        GPT_eval_5_metric(dial_gen)
    save_json(data, './interact_output/simulator_t5_large_data3.0_and_system_gpt/simulator_t5_large_data3.0_and_system_gpt-3.5-turbo_withscore.json')



    # 计算 gpt 打分和人工打分的相似性
    # gpt_eval_score = []  # 总的平均分：2.297917  positive平均分：2.412500  negative平均分：2.183333
    # human_score = []  # 总的平均分：2.575000  positive平均分：2.691667  negative平均分：2.458333
    #
    # eval_datas = load_json('./interact_output/t5_large_us15_blenderbot_small_generate_results.json')
    # human_datas = load_json('./data/empathetic_dialogues/ieval_data.json')
    #
    # # for eval_data_idx in range(0, len(eval_datas), 2):
    # for obj in human_datas:
    #     ieval_positive_conv_id = obj['positive']['conv_id']
    #     # print(obj['positive'])
    #     blenderbot_positive_human_score = obj['positive']['purple']['rating']
    #
    #     ieval_negative_conv_id = obj['negative']['conv_id']
    #     blenderbot_negative_human_score = obj['negative']['purple']['rating']
    #
    #     # for eval_data in eval_datas:
    #     #     if eval_data['dialog_id'] == ieval_positive_conv_id:
    #     #         if eval_data['gpt-3.5-turbo_score'] == 0:
    #     #             eval_data['gpt-3.5-turbo_score'] = 1
    #     #         elif eval_data['gpt-3.5-turbo_score'] == 0.5:
    #     #             eval_data['gpt-3.5-turbo_score'] = 2
    #     #         else:
    #     #             eval_data['gpt-3.5-turbo_score'] = 3
    #     #
    #     #         gpt_eval_score.append(eval_data['gpt-3.5-turbo_score'])
    #     #         human_score.append(blenderbot_positive_human_score)
    #     #         break
    #
    #     for eval_data in eval_datas:
    #         if eval_data['dialog_id'] == ieval_negative_conv_id:
    #             if eval_data['gpt-3.5-turbo_score'] == 0:
    #                 eval_data['gpt-3.5-turbo_score'] = 1
    #             elif eval_data['gpt-3.5-turbo_score'] == 0.5:
    #                 eval_data['gpt-3.5-turbo_score'] = 2
    #             else:
    #                 eval_data['gpt-3.5-turbo_score'] = 3
    #
    #             gpt_eval_score.append(eval_data['gpt-3.5-turbo_score'])
    #             human_score.append(blenderbot_negative_human_score)
    #             break
    #
    # gpt_eval_score = [2.297917, 2.412500, 2.183333]
    # human_score = [2.575000, 2.691667, 2.458333]
    # print(calculate_pearson(gpt_eval_score, human_score))


