# # from transformers import T5Tokenizer
# #
# # tokenizer = T5Tokenizer.from_pretrained('t5-small')
# #
# #
# # encoded_text = tokenizer.encode("hello, my name is lee . ")
# # print(encoded_text)
# # print(tokenizer.decode(encoded_text))
# #
# # 这种写法可以看看 self.bs_prefix_id = self.dialog_tokenizer.convert_tokens_to_ids(self.dialog_tokenizer.tokenize(bs_prefix_text))
# #
# # # 明天看一下源码MultiWOZIterator类的get_data_iterator_ds和get_data_iterator_us和get_batches方法
# # # 改到EDIterator这里了，然后再改一下EDReader就好了
# # # EDReader就是把数据读出来，按照原数据组织一下ED数据集就好了，目标格式已经定义好了(data_sample.json)，下一个任务就把ED转为目标格式
# # # 格式已经改好了，然后就是编码数据就好了
# # # 数据编码也完成了，test.jsonl文件已经ok了，明天再改一改小bug，然后把train.jsonl和valid.jsonl也编码一下，然后把我写的代码移植到原来的代码里
# # # ERReader，数据处理部分全部完成了
#
# from transformers import T5Tokenizer
# def init_tokenizer():
#     tokenizer = T5Tokenizer.from_pretrained('t5-small')
#
#     # # special_tokens 可以去掉，或者改成自己需要的
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
#
# tokenizer = init_tokenizer()
#
# # encoded_text = tokenizer.tokenize("<bos_user> it's mine. hello, my name is lee. <eos_user> <bos_resp> <eos_resp> <bos_goal> <eos_goal> [emotion] [situation]")
# # print(encoded_text)
# # encoded_text = tokenizer.tokenize("<bos_user> it 's mine. hello, my name is lee. <eos_user> <bos_resp> <eos_resp> <bos_goal> <eos_goal> [emotion] [situation]")
# # print(encoded_text)
# # # print(tokenizer.decode(5))
#
# def encode_text(text, bos_token=None, eos_token=None):
#     tokens = text.split() if isinstance(text, str) else text
#
#     assert isinstance(tokens, list)
#
#     if bos_token is not None:
#         if isinstance(bos_token, str):
#             bos_token = [bos_token]
#
#         tokens = bos_token + tokens
#
#     if eos_token is not None:
#         if isinstance(eos_token, str):
#             eos_token = [eos_token]
#
#         tokens = tokens + eos_token
#
#     # <MOD>
#     # encoded_text = tokens
#     encoded_text = tokenizer.encode(" ".join(tokens))
#
#     # except eos token
#     if encoded_text[-1] == tokenizer.eos_token_id:
#         encoded_text = encoded_text[:-1]
#
#     return encoded_text
#
#
# from utils.utils import load_json_by_line, save_json, load_pickle, save_pickle, get_or_create_logger
# from reader import EDReader
# from config import get_config
# import os
# import json
#
# def encode_data(data_type):
#     data = load_json_by_line(os.path.join("data/empathetic_dialogues", "{}.jsonl".format(data_type)))
#
#     encoded_data = []
#     convs = []
#     dial = []
#
#     # 把每个dial整合到一起
#     for idx, line in enumerate(data):
#         dial.append(line)
#         # 清空前保存dial
#         if idx + 1 == len(data) or line['conv_idx'] != data[idx + 1]['conv_idx']:
#             convs.append(dial)
#             dial = []
#
#     # 定义特殊token
#     bos_user_token = '<bos_user>'
#     eos_user_token = '<eos_user>'
#
#     bos_resp_token = '<bos_resp>'
#     eos_resp_token = '<eos_resp>'
#
#     bos_goal_token = '<bos_goal>'
#     eos_goal_token = '<eos_goal>'
#
#     # 调整格式并编码
#     new_convs = []
#     for dial_id, dial in enumerate(convs):
#         new_dial = []
#         len_dial = len(dial)
#         # 如果不是偶数个对话语句的时候去掉最后一句
#         if len_dial % 2 != 0:
#             len_dial -= 1
#
#         for idx in range(0, len_dial, 2):
#             # 把两句话（一轮对话）整合到一起
#             turn = {}
#             turn['dial_id'] = data_type + "_" + str(dial_id)
#             turn['turn_num'] = int(idx / 2)
#             turn['user'] = encode_text(dial[idx]['utterance'], bos_token=bos_user_token, eos_token=eos_user_token)
#             turn['resp'] = encode_text(dial[idx + 1]['utterance'], bos_token=bos_resp_token, eos_token=eos_resp_token)
#             turn['goal_state'] = "[emotion] " + dial[idx]['emotion'] + " [situation] " + dial[idx]['situation']
#             turn['goal_state'] = encode_text(turn['goal_state'], bos_token=bos_goal_token, eos_token=eos_goal_token)
#             new_dial.append(turn)
#         new_convs.append(new_dial)
#
#     return new_convs
#
#
# train = encode_data("train")
# dev = encode_data("valid")
# test = encode_data("test")
#
# data = {"train": train, "dev": dev, "test": test}
# save_json(data, "data/empathetic_dialogues/data_processed.json")
#
#
# # # data/empathetic_dialogues/test.jsonl
# # data_type = "test"
# # data = load_json_by_line(os.path.join("data/empathetic_dialogues", "{}.jsonl".format(data_type)))
# # print(len(data))
# #
# # # cfg = get_config()
# # # reader = EDReader(cfg)
# #
# # encoded_data = []
# # convs = []
# # # count_conv = 0
# # dial = []
# # for idx, line in enumerate(data):
# #     dial.append(line)
# #     # 清空前保存dial
# #     if idx + 1 == len(data) or line['conv_idx'] != data[idx + 1]['conv_idx']:
# #         convs.append(dial)
# #         dial = []
# # # 把每个dial整合到一起
# # # save_json(convs, "data/empathetic_dialogues/test_processed.json")
# #
# #
# # # 定义特殊token
# # bos_user_token = '<bos_user>'
# # eos_user_token = '<eos_user>'
# #
# # bos_resp_token = '<bos_resp>'
# # eos_resp_token = '<eos_resp>'
# #
# # bos_goal_token = '<bos_goal>'
# # eos_goal_token = '<eos_goal>'
# #
# # new_convs = []
# # for dial_id, dial in enumerate(convs):
# #     new_dial = []
# #     for idx in range(0, len(dial), 2):  # <HERE> 这里明天要改一下，如果不是偶数个对话语句的时候要特殊处理
# #         # 把两句话（一轮对话）整合到一起
# #         turn = {}
# #         turn['dial_id'] = data_type + "_" + str(dial_id)
# #         turn['turn_num'] = int(idx / 2)
# #         turn['user'] = encode_text(dial[idx]['utterance'], bos_token=bos_user_token, eos_token=eos_user_token)
# #         turn['resp'] = encode_text(dial[idx + 1]['utterance'], bos_token=bos_resp_token, eos_token=eos_resp_token)
# #         turn['goal_state'] = "[emotion] " + dial[idx]['emotion'] + " [situation] " + dial[idx]['situation']
# #         turn['goal_state'] = encode_text(turn['goal_state'], bos_token=bos_goal_token, eos_token=eos_goal_token)
# #         new_dial.append(turn)
# #     new_convs.append(new_dial)
# # # 把每个dial中的每个turn整合到一起
# # data = {}
# # data['train'] = new_convs
# # save_json(data, "data/empathetic_dialogues/test_processed.json")
#
#
#

# 上面是数据处理的代码，告一段落了



'''
新的任务，测试 EDIterator 部分
EDIterator主要函数：get_batches(BaseIterator.get_batches), get_data_iterator***(get_data_iterator_us, get_data_iterator_ds)
get_batches不用修改，get_data_iterator_us 和 get_data_iterator_ds 修改完了
EDIterator主要函数测试完毕，没有问题
'''

# from utils.utils import load_json_by_line, save_json, load_pickle, save_pickle, get_or_create_logger
# from reader import EDReader, EDIterator
# from config import get_config
# import os
# import json
# import torch
#
# cfg = get_config()
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)
#
# setattr(cfg, "device", device)
# setattr(cfg, "num_gpus", num_gpus)
#
# reader = EDReader(cfg)
# iterator = EDIterator(reader)
#
# train_batches, _, _, _ = iterator.get_batches("train", cfg.batch_size,
#                                               cfg.num_gpus, shuffle=True,
#                                               num_dialogs=2,  # cfg.num_train_dialogs,
#                                               excluded_domains=cfg.excluded_domains)
# print(train_batches)
#
#
# # 对话系统的训练数据和用户模拟器的训练数据的区别
# get_data_iterator_ds = iterator.get_data_iterator('ds')
# get_data_iterator_us = iterator.get_data_iterator('us')
# train_ds_iterator = get_data_iterator_ds(train_batches, cfg.ururu, cfg.context_size)
# train_us_iterator = get_data_iterator_us(train_batches, cfg.ururu, cfg.context_size)
# # from transformers import T5Tokenizer
# # tokenizer = T5Tokenizer.from_pretrained('t5-small')
# print('--------------对话系统的训练数据-----------------')
# for step, batch in enumerate(train_ds_iterator):
#     inputs, resp_labels = batch
#     print('inputs----------------')
#     inputs = reader.tokenizer.decode(inputs[0])
#     print(inputs)
#     print('resp_labels----------------')
#     resp_labels = reader.tokenizer.decode(resp_labels[0])
#     print(resp_labels)
# print('--------------用户模拟器的训练数据-----------------')
# for step, batch in enumerate(train_us_iterator):
#     inputs, resp_labels = batch
#     print('inputs----------------')
#     inputs = reader.tokenizer.decode(inputs[0])
#     print(inputs)
#     print('resp_labels----------------')
#     resp_labels = reader.tokenizer.decode(resp_labels[0])
#     print(resp_labels)

'''
新的任务，测试 EDRunner 部分
EDRunner.train()已经没问题啦
validation等其他细节还要修改
predict等部分还未完成，先改interact.py
'''

# from runner import EDRunner
# from utils.utils import load_json_by_line, save_json, load_pickle, save_pickle, get_or_create_logger
# from reader import EDReader, EDIterator
# from config import get_config
# import os
# import json
# import torch
#
# cfg = get_config()
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)
#
# setattr(cfg, "device", device)
# setattr(cfg, "num_gpus", num_gpus)
#
# runner = EDRunner(cfg)
# runner.train()  # ds epoch6最好; us epoch4最好


'''
interact.py
主要函数：generate_single_dialog, train_RL
train_RL部分稍后再改，先改generate_single_dialog

生成的对话格式：
[
    {"sng0073.json": goal,
     "terminate_reason": "goal清空后终止",
     "log": 对话历史
     },
    {"sng0073.json": goal,
     "terminate_reason": "goal清空后终止",
     "log": 对话历史
     }
]

'''
# from runner import EDRunner
# from utils.utils import load_json_by_line, load_json, save_json, load_pickle, save_pickle, get_or_create_logger
# from reader import EDReader, EDIterator
# from config import get_config
# import os
# import json
# import torch
# from transformers import T5ForConditionalGeneration
#
# logger = get_or_create_logger(__name__)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# def generate_single_dialog(user_goal):
#     simulator_model = T5ForConditionalGeneration.from_pretrained('./simulator_t5_small/ckpt-epoch11')
#     dialog_model = T5ForConditionalGeneration.from_pretrained('./dialogue_t5_small/ckpt-epoch11')
#     simulator_tokenizer = T5ForConditionalGeneration.from_pretrained('./simulator_t5_small/ckpt-epoch11')
#     dialog_tokenizer = T5ForConditionalGeneration.from_pretrained('./dialogue_t5_small/ckpt-epoch11')
#
#     simulator_model.to(device)
#     dialog_model.to(device)
#
#     dial_gen = {user_goal['dialog_id']: {'goal': user_goal['goal']}}
#
#
#
# goal_list = load_json('data/empathetic_dialogues/goal_list.json')
# goal = goal_list['test'][0]
# dial_gen = generate_single_dialog(goal)

'''
今天看完了 Approximating Online Human Evaluation of Social Chatbots with Prompting 这篇，感觉一般，github给了数据但是没给代码，不过代码应该很简单，可以自己复现。
有一些做法和流程可以借鉴。
然后要超过他的相关性和用户模拟器的难度还挺大的。
后面要完善用户模拟器的生成效果（比如可以加prompt给t5，应该好很多，然后用t5-base和t5-large试试）
完成这部分之后就只差强化学习了

网连不上，算了
'''
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import os
# # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7891"
# # os.environ["HTTPS_PROXY"] = "https://127.0.0.1:7891"
# # os.environ["FTP_PROXY"] = "http://127.0.0.1:7891"
# # os.environ["ALL_PROXY"] = "http://127.0.0.1:7891"
# # os.environ["NO_PROXY"] = "127.0.0.1,localhost"
# tokenizer = T5Tokenizer.from_pretrained('t5-large')
# model = T5ForConditionalGeneration.from_pretrained('t5-large')


'''
有两个任务：1.加上bleu指标 2.完善tensorboard的数据显式 3.ckpt只保存bestbleu，bestloss，latest
然后先用small模型试一下
2完成了，再看看1
1的流程和代码知道了，然后改掉了predict部分的代码，后面在 validation 函数里加上bleu指标的计算就可以了，再加个 tensorboard 的写入。

us的bleu有15.51，ds的bleu只有0.57，怎么会这样，要解决一下
观察后发现，虽然us的bleu高，但是只是因为goal_state直接给了us模型对话信息；ds的bleu虽然低，但是反而通顺很多；并且us和ds有一个共同的缺点，重复性很高，怎么办？

目前的想法：
1.再研究研究T5，然后加个任务前缀试试
2.自己加的 special_tokens 可以尝试换成T5预留的 special_tokens <extra_id_0>.. 等等
3.goal_state给了us模型偷懒的机会，尝试可不可以把这一部分先编码再加进入
4.增加温度

main.py训练的参数列表
python main.py -agent_type us -run_type train -backbone t5-small -model_dir simulator_t5_test -epoch 1
python main.py -data_version 3.0 -agent_type us -run_type train -backbone t5-large -model_dir simulator_t5_large_data3.0 -epoch 10
main.py推理的参数列表
python main.py -run_type predict -predict_agent_type us -pred_data_type test -ckpt ./simulator_t5_small_lr1e-4_bs8/ckpt-epoch5 -output inference.json -batch_size 16
python main.py -data_version 3.0 -run_type predict -predict_agent_type us -pred_data_type test -ckpt ./simulator_t5_large_data3.0_spancutoff/ckpt-epoch10 -output inference.json -batch_size 64
python main.py -data_version 3.0 -run_type predict -predict_agent_type ds -pred_data_type test -ckpt ./interact_model/simulator_t5_large_data3.0_interact_09/ckpt-epoch1 -output inference.json -batch_size 8
'''
# import nltk
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# # 设置平滑函数
# smoother = SmoothingFunction()
#
# # 参考答案列表，通常有多个参考答案
# references = [["this", "is", "an", "answer"], ["hello", "world"]]
#
# # 生成的文本，一个句子
# candidate = ["this", "is", "a", "test"]
#
# # 计算BLEU-2分数
# score_bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5), smoothing_function=smoother.method0)
#
# # 计算BLEU-3分数
# score_bleu3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33), smoothing_function=smoother.method0)
#
# # 计算BLEU-4分数
# score_bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method0)
#
# # 打印不同n-gram的BLEU分数
# print("BLEU-2 Score:", score_bleu2)
# print("BLEU-3 Score:", score_bleu3)
# print("BLEU-4 Score:", score_bleu4)
#
#
# from utils.utils import load_json, save_json
#
# # results = load_json("simulator_t5_small_lr1e-4_bs8/ckpt-epoch5/inference.json")
# results = load_json("./simulator/simulator_t5_large_data3.0/simulator_rl_4_epoch_4/inference.json")
#
# def calculate_bleu(results, agent_type):
#     score_bleu1 = 0
#     score_bleu2 = 0
#     score_bleu3 = 0
#     score_bleu4 = 0
#     count_turns = 0
#     smoother = SmoothingFunction()
#     for id, dial in results.items():
#         for turn in dial:
#             count_turns += 1
#             if agent_type == 'us':
#                 references = [turn['user'].split()[1:-1]]
#                 candidate = turn['user_gen'].split()
#             elif agent_type == 'ds':
#                 references = [turn['resp'].split()[1:-1]]
#                 candidate = turn['sys_gen'].split()
#
#             score_bleu1 += sentence_bleu(references, candidate, weights=(1,), smoothing_function=smoother.method0)
#             score_bleu2 += sentence_bleu(references, candidate, weights=(0.5, 0.5), smoothing_function=smoother.method0)
#             score_bleu3 += sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33), smoothing_function=smoother.method0)
#             score_bleu4 += sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method0)
#
#
#             # turn['user'] = ' '.join(turn['user'].split()[1:-1])
#             # turn['resp'] = ' '.join(turn['resp'].split()[1:-1])
#     # print("BLEU-4 Score: {:.2f}".format(score_bleu4 / count_turns * 100))
#     return score_bleu1 / count_turns * 100, score_bleu2 / count_turns * 100, score_bleu3 / count_turns * 100, score_bleu4 / count_turns * 100
#
# a, b, c, d = calculate_bleu(results, 'us')
# print(a)
# print(b)
# print(c)
# print(d)

'''
计算三个实验的bleu
'''
# from utils.utils import load_json
# results = load_json('./simulator_t5_large_data3.0/ckpt-epoch10/data1.0_inference.json')
# score_bleu1, score_bleu2, score_bleu3, score_bleu4 = calculate_bleu(results, 'us')
# print("BLEU-1 Score: {:.2f}".format(score_bleu1))
# print("BLEU-2 Score: {:.2f}".format(score_bleu2))
# print("BLEU-3 Score: {:.2f}".format(score_bleu3))
# print("BLEU-4 Score: {:.2f}".format(score_bleu4))



'''
测一测t5-large训练的us
先把us的dev tensorboard补充完整
上面没搞

先测一下两个small的bleu，然后挑出最好的us和ds模型做交互，看看效果，估计和之前看到的一样
然后测base的，然后测large的，估计要两天

python test.py -run_type predict -predict_agent_type us -pred_data_type test -ckpt ./simulator_t5_small -batch_size 8

下面是做推理的代码，用上面的命令行运行
'''
# from tensorboardX import SummaryWriter
# import os
#
# model_dir = "simulator_t5_small_test"
# summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))
# summary_writer.add_scalar("dev/bleu", 10, global_step=1)
# summary_writer.add_scalar("dev/bleu", 20, global_step=2)
# summary_writer.add_scalar("dev/bleu", 30, global_step=3)
# summary_writer.add_scalar("dev/bleu", 40, global_step=4)
# summary_writer.add_scalar("dev/bleu", 50, global_step=5)
#
# summary_writer.add_scalar("dev/loss", 10, global_step=1)
# summary_writer.add_scalar("dev/loss", 20, global_step=2)
# summary_writer.add_scalar("dev/loss", 30, global_step=3)
# summary_writer.add_scalar("dev/loss", 40, global_step=4)
# summary_writer.add_scalar("dev/loss", 50, global_step=5)
#
# summary_writer.add_scalar("dev/acc", 10, global_step=1)
# summary_writer.add_scalar("dev/acc", 20, global_step=2)
# summary_writer.add_scalar("dev/acc", 30, global_step=3)
# summary_writer.add_scalar("dev/acc", 40, global_step=4)
# summary_writer.add_scalar("dev/acc", 50, global_step=5)


# import random
# import os
#
# import torch
# import numpy as np
#
# from config import get_config
# from runner import MultiWOZRunner, EDRunner
# from tensorboardX import SummaryWriter
# from utils.utils import get_or_create_logger, calculate_bleu, save_json
#
# logger = get_or_create_logger(__name__)
#
#
# def all_predict(trained_model, start_epoch=1, end_epoch=6):
#     directory_path = os.path.join(os.path.join(trained_model, "inference", "tensorboard"))
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path, exist_ok=True)
#
#     summary_writer = SummaryWriter(os.path.join(trained_model, "inference", "tensorboard"))
#
#     cfg = get_config()
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)
#
#     setattr(cfg, "device", device)
#     setattr(cfg, "num_gpus", num_gpus)
#
#     logger.info("Device: %s (the number of GPUs: %d)", str(device), num_gpus)
#
#     if cfg.seed > 0:
#         random.seed(cfg.seed)
#         np.random.seed(cfg.seed)
#         torch.manual_seed(cfg.seed)
#         torch.cuda.manual_seed_all(cfg.seed)
#
#         logger.info("Set random seed to %d", cfg.seed)
#
#     for epoch in range(start_epoch, end_epoch + 1):
#         cfg.ckpt = os.path.join(trained_model, "ckpt-epoch{}".format(epoch))
#
#         runner = EDRunner(cfg)  # runner = MultiWOZRunner(cfg)
#
#         bleu = 0
#         results = None
#         if cfg.predict_agent_type == 'ds':
#             results = runner.predict()
#             bleu = calculate_bleu(results, cfg.predict_agent_type)
#         elif cfg.predict_agent_type == 'us':
#             results = runner.us_predict()
#             bleu = calculate_bleu(results, cfg.predict_agent_type)
#         logger.info(" ".join(["[Test]", f"ckpt-epoch{epoch}", "BLEU-4 Score: {:.2f}".format(bleu)]))
#
#         summary_writer.add_scalar("test_bleu", bleu, global_step=epoch)
#
#         save_json(results, os.path.join(trained_model, "inference", "ckpt-epoch{}_inference.json".format(epoch)))
#
#
# trained_model = "./dialogue_t5_small"
# all_predict(trained_model)


'''
把生成的对话转化成可以评分的形式
I am a Speaker, feeling <emotion> because <situation>. I shared these emotions with a Listener in a
dialog, expecting empathy and understanding from
them. Our dialog went as follows.
Speaker: <LLM’s input #1>
Listener: <Bot’s response #1>
Speaker: <LLM’s input #2>
Listener: <Bot’s response #2>
Speaker: <LLM’s input #3>
Listener: <Bot’s response #3>
In such open-ended dialogs, good listeners demonstrate coherence and maintain a good conversation flow, they display a likeable personality and understanding ofthe speaker. On the contrary, bad listeners don’t follow the context and don't show much interest in the conversation. I would rate the Listener in my dia-
log as ___, choosing from Bad, Okay, and Good
options.
'''

# from utils.utils import get_or_create_logger, calculate_bleu, save_json, load_json
# import random
# from tqdm import tqdm
#
# file_path = 'interact_output/t5_large_us6_ds6_generate_results.json'
# convs = load_json(file_path)
#
# random_numbers = [random.randint(0, 99) for _ in range(5)]
#
# for idx in tqdm(random_numbers):
#     goal = None
#     for k, v in convs[idx].items():
#         goal = v['goal']
#         break
#     goal = goal.split()
#     emotion = goal[1]
#     situation = ' '.join(goal[3:])
#     LLMinput1 = convs[idx]['log'][0]['user']
#     Botresponse1 = convs[idx]['log'][0]['sys']
#     LLMinput2 = convs[idx]['log'][1]['user']
#     Botresponse2 = convs[idx]['log'][1]['sys']
#     LLMinput3 = convs[idx]['log'][2]['user']
#     Botresponse3 = convs[idx]['log'][2]['sys']
#     prompt = f'''
# I am a Speaker, feeling {emotion} because {situation}. I shared these emotions with a Listener in a dialog, expecting empathy and understanding from them. Our dialog went as follows.
# Speaker: {LLMinput1}
# Listener: {Botresponse1}
# Speaker: {LLMinput2}
# Listener: {Botresponse2}
# Speaker: {LLMinput3}
# Listener: {Botresponse3}
# In such open-ended dialogs, good listeners demonstrate coherence and maintain a good conversation flow, they display a likeable personality and understanding ofthe speaker. On the contrary, bad listeners don't follow the context and don't show much interest in the conversation. I would rate the Listener in my dialog as ___, choosing from Bad, Okay, and Good options.
# '''
#     print(prompt)




'''
把 ieval_data 的数据整理分离出来
未完成
'''

# from utils.utils import get_or_create_logger, calculate_bleu, save_json, load_json
#
# data_path = 'data/empathetic_dialogues/ieval_data.json'
# ieval_data = load_json(data_path)
#
# goal_list = []
#
# for obj in ieval_data:
#     dialog_id = obj['positive']['conv_id']
#     emotion = obj['positive']['emotion']
#     situation = obj['positive']['prompt']
#     goal = "[emotion] " + emotion + " [situation] " + situation
#
#     goal_list.append({'dialog_id': dialog_id, 'goal': goal})


'''
续写对话
把json转prompt

用API续写对话的代码 和 数据处理的代码 在OpenAI_API_test的项目里
最终得到 train_gpt3_5.jsonl 等文件

I am a Speaker, feeling disgusted because I saw a huge cockroach outside my house today. We live in Texas so they are common but still gross! I shared these emotions with a Listener in a dialog, expecting empathy and understanding from them. Our dialog went as follows.

Speaker: I saw a huge cockroach outside my house today!
Listener: did you call the exterminator?
Speaker: Not yet since it's the weekend. We live in Texas so they are common but still gross! I'm glad I haven't see any in my house.
Listener: I live in Texas to so i know those feels.

I hope you can help me continue writing 6 turns of the dialogue. The Listener's response of the continuation dialogue should display a likeable personality and understanding of the speaker.
'''
# import json
# from tqdm import tqdm
# import random
# from utils.utils import save_json
#
# def load_json_by_line(load_path, lower=True):
#     with open(load_path, "r", encoding="utf-8") as f:
#         data = []
#         for line in f.readlines():
#             if lower:
#                 line = line.lower()
#             data.append(json.loads(line))
#
#         return data
#
#
# file_name = 'test'
#
# file_path = f'data/empathetic_dialogues/{file_name}.jsonl'
# data = load_json_by_line(file_path, lower=False)
#
# convs = []
# dial = []
#
# # 把每个dial整合到一起
# for idx, line in enumerate(data):
#     dial.append(line)
#     # 清空前保存dial
#     if idx + 1 == len(data) or line['conv_idx'] != data[idx + 1]['conv_idx']:
#         convs.append(dial)
#         dial = []
#
# # 调整为prompt格式
# new_convs = []
# for dial_id, dial in enumerate(convs):
#     emotion = dial[0]['emotion']
#     situation = dial[0]['situation']
#     prompt = f'''I am a Speaker, feeling {emotion} because {situation}. I shared these emotions with a Listener in a dialog, expecting empathy and understanding from them. Our dialog went as follows.
#
# '''
#
#     len_dial = len(dial)
#     # 如果不是偶数个对话语句的时候去掉最后一句
#     if len_dial % 2 != 0:
#         len_dial -= 1
#
#     new_dial = {}
#     utterances = []
#
#     # 将1个对话转成prompt
#     for idx in range(0, len_dial, 2):
#         speaker_input = dial[idx]['utterance']
#         listener_input = dial[idx + 1]['utterance']
#
#         utterances.append(speaker_input)
#         utterances.append(listener_input)
#
#         prompt += f"Speaker: {speaker_input}\n"
#         prompt += f"Listener: {listener_input}\n"
#
#     prompt += '''
# I hope you can help me continue writing 4 turns of the dialogue. The Listener's response of the continuation dialogue should display a likeable personality and understanding of the speaker.'''
#     new_dial['conv_idx'] = dial[0]['conv_idx']
#     new_dial['emotion'] = dial[0]['emotion']
#     new_dial['situation'] = dial[0]['situation']
#     new_dial['prompt'] = prompt
#     new_dial['dialogue'] = utterances
#     new_dial['continue_dialogue'] = ""
#
#     new_convs.append(new_dial)
#
# # random_numbers = [random.randint(0, len(new_convs) - 1) for _ in range(5)]
# #
# # for idx in tqdm(random_numbers):
# #     print(new_convs[idx]['prompt'])
# #     print('----------------------')
#
# save_json(new_convs, f'data/empathetic_dialogues/{file_name}_prompt_gpt3_5_brief.jsonl')


'''
比较t5-small在不同数据集上的生成效果
'''
# from utils.utils import load_json
#
# v1_data_small = load_json('interact_output/v1_us4_ds6_generate_results.json')
# v2_data_small = load_json('interact_output/t5_small_datagpt3_5_us10_ds10_generate_results.json')
# v1_data_large = load_json('interact_output/t5_large_us6_ds6_generate_results.json')
#
# for i in v2_data_small:
#     for j in v1_data_large:
#         for k in v1_data_small:
#             if list(i.keys())[0] == list(j.keys())[0] and list(i.keys())[0] == list(k.keys())[0]:
#                 print(list(i.keys())[0])

'''
1. reward函数怎么写得，reward怎么获得的
2. rl_loss是什么
3. reward和rl_loss有什么联系吗
4. update怎么更新的
5. evaluator.py要看一下
前4个完成，第5个就是评测文件，先不看了

1. user_prob 和 sys_prob 搞一下
2. openai api 接进来评测

python 3.7.2
pytorch 1.10.1+cu111
openai

[INFO] Total training steps = 150840, warmup steps = 30168

强化学习训练
python interact_t5.py -do_rl_training -seed 1998 -simulator_save_path simulator_rl_test -dialog_save_path dialog_rl_test

两个模型交互
python interact_t5.py -simulator_path ./simulator_t5_small_datagpt3_5/ckpt-epoch10 -dialog_sys_path ./dialogue_t5_small_datagpt3_5/ckpt-epoch10 -model_name t5-small -generate_results_path interact_output/t5_small_datagpt3_5_us10_ds10_generate_results.json

1.续写要弄完
2.续写数据集上跑一个版本
3.续写+数据增强跑一个版本
4.奖励的prompt要改
5.代码里的数据读入要改，改成version控制就可以
'''



'''
数据增强测试
'''
# import torch
#
# def generate_token_cutoff_embedding(embeds, input_lens, aug_cutoff_ratio):
#     input_embeds = []
#     for i in range(embeds.shape[0]):
#         cutoff_length = int(input_lens * aug_cutoff_ratio)
#         zero_index = torch.randint(input_lens, (cutoff_length,))
#
#         cutoff_embed = embeds[i]
#         tmp_mask = torch.ones(cutoff_embed.shape[0], ).to(embeds.device)
#         for ind in zero_index:
#             tmp_mask[ind] = 0
#
#         cutoff_embed = torch.mul(tmp_mask[:, None], cutoff_embed)
#
#         input_embeds.append(cutoff_embed)
#
#     input_embeds = torch.stack(input_embeds, dim=0)
#
#     return input_embeds
#
#
# def generate_span_cutoff_embedding(embeds, input_lens, aug_cutoff_ratio):
#     input_embeds = []
#     for i in range(embeds.shape[0]):
#         cutoff_length = int(input_lens * aug_cutoff_ratio)
#         start = int(torch.rand(1) * (input_lens - cutoff_length))
#         # print(input_lens[i], cutoff_length, start)
#         cutoff_embed = torch.cat((embeds[i][:start],
#                                   torch.zeros([cutoff_length, embeds.shape[-1]],
#                                               dtype=torch.float).to(embeds.device),
#                                   embeds[i][start + cutoff_length:]), dim=0)
#         input_embeds.append(cutoff_embed)
#     input_embeds = torch.stack(input_embeds, dim=0)
#     return input_embeds
#
# # x = torch.tensor([[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], [[4.0, 4.1], [5.0, 5.1], [6.0, 6.1]]])
# # x = torch.rand(2, 10, 3)
# # print(x.size())
# # print(x)
# # _, seq_len, _ = x.size()
# # x = generate_token_cutoff_embedding(x, seq_len, aug_cutoff_ratio=0.5)
# # print(x.size())
# # print(x)
#
#
# from transformers import T5ForConditionalGeneration, AutoTokenizer
# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer = AutoTokenizer.from_pretrained('t5-small')
#
#
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
#
# # x = torch.tensor([0, 1, 2, 3, 4])
# # x = model.embeddings(x)
# # print(x)
# input_ids = tokenizer("how are you", return_tensors="pt").input_ids
# labels = tokenizer("i am fine", return_tensors="pt").input_ids
# # print(input_ids)
# embedding = model.shared(input_ids)
# # print(embedding)
# # print(model.shared(input_ids).size())
#
# _, seq_len, _ = embedding.size()
# embedding = generate_span_cutoff_embedding(embedding, seq_len, aug_cutoff_ratio=0)
# outputs = model(inputs_embeds=embedding, labels=labels)
# # outputs = model(input_ids=input_ids, labels=labels)
#
# loss = outputs.loss
# logits = outputs.logits
#
# print(loss, logits)


'''
1.下载blenderbot的模型，并简单测试  --差不多完成了
2.把ieval和ED数据集整合整理好  --这个不用搞，直接用ieval_data.json里面的信息就可以
3.用blenderbot和us交互，生成一系列对话（使用ieval里的测试数据） --完成
4.将生成的对话交给chatgpt打分  --完成
5.将得到的评分和人类打分比较，计算相似度  --完成
'''

# from transformers import BlenderbotSmallForConditionalGeneration, AutoTokenizer
# dialog_model = BlenderbotSmallForConditionalGeneration.from_pretrained('blenderbot_small-90M')
# dialog_tokenizer = AutoTokenizer.from_pretrained('blenderbot_small-90M')

# input_ids =
# dialog_generate = dialog_model.generate.__wrapped__
# model_output = dialog_generate(
#     dialog_model,
#     input_ids=input_ids,
#     # decoder_input_ids=bspn_decoder_input_ids,
#     eos_token_id=dialog_tokenizer.eos_token_id,
#     max_length=100,
#     # max_length=80,
# )

# from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration
#
# mname = "blenderbot_small-90M"
# model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
# tokenizer = AutoTokenizer.from_pretrained(mname)

# special_tokens = []
#
# special_tokens.append("<bos_user>")
# special_tokens.append("<eos_user>")
# special_tokens.append("<bos_resp>")
# special_tokens.append("<eos_resp>")
# special_tokens.append("<bos_goal>")
# special_tokens.append("<eos_goal>")
# special_tokens.append("[emotion]")
# special_tokens.append("[situation]")
#
# tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
#
# model.resize_token_embeddings(len(tokenizer))
# print(len(tokenizer))


# UTTERANCE = "My friends are cool but they eat too many carbs."
# print("Human: ", UTTERANCE)  # Human:  My friends are cool but they eat too many carbs.
# inputs = tokenizer([UTTERANCE], return_tensors="pt")
# reply_ids = model.generate(**inputs)
# print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])  # Bot:  what kind of carbs do they eat? i don't know much about carbs.
# REPLY = "I'm not sure"
# print("Human: ", REPLY)  # Human: I'm not sure
# NEXT_UTTERANCE = "My friends are cool but they eat too many carbs. </s> <s> what kind of carbs do they eat? i don't know much about carbs </s> <s> I'm not sure."
# # ("My friends are cool but they eat too many carbs.</s> <s>what kind of carbs do they eat? "
# # "i don't know much about carbs</s> "
# # "<s> I'm not sure."
# # )
# # print(type(NEXT_UTTERANCE))
# # print(len(NEXT_UTTERANCE))
# inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
# print(inputs)
# next_reply_ids = model.generate(**inputs)
# print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])  # Bot:  they eat a lot of carbs. carbs are high in fat, protein, and carbohydrates.


# UTTERANCE = "My friends are cool but they eat too many carbs."
# inputs = tokenizer([UTTERANCE], return_tensors="pt")
# reply_ids = model.generate(**inputs)
# print(type(tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]))
# print(tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
#
# NEXT_UTTERANCE = "My friends are cool but they eat too many carbs. </s> <s> what kind of carbs do they eat? i don't know much about carbs </s> <s> I'm not sure."
# inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
# next_reply_ids = model.generate(**inputs)
# print(type(tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0]))
# print(tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])


'''
计算Pearson相关系数
'''
# import pandas as pd
# # import scipy.stats as stats
#
# # 构造数据
# data = {
#     'Study Hours': [2, 4, 6, 8, 10, 3, 5, 7, 9, 11],
#     'Test Scores': [60, 70, 80, 85, 90, 65, 75, 82, 88, 92]
# }
# df = pd.DataFrame(data)
#
# # 计算皮尔逊相关系数
# pearson_corr = df.corr(method='pearson')
# print("皮尔逊相关系数:\n", pearson_corr)

# # 另一种方法直接使用scipy
# pearson_corr_value, _ = stats.pearsonr(df['Study Hours'], df['Test Scores'])
# print("皮尔逊相关系数值:", pearson_corr_value)


'''
任务目标
用大模型检测错误并优化模型
任务流程
1.用us和ds交互生成n个对话  --完成
2.将生成n个对话整理成prompt的格式  --完成
3.用大模型对这n个对话进行错误检测  --完成
4.将修改后的对话整理成
4.将修改后的对话整理成train_demo_batches.json格式  --
5.对模型进行优化

可以复用之前的训练代码，从 runner.iterator.get_batches 后面直接用就可以了
监督训练成本太高了，可以选取每种情绪100个例子（总共3200）个例子进行训练
修订前的对话需要有一定的质量，不然修订后效果也不好
'''


'''
用大模型比较 t5_large_us10_ds8 和 t5_large_rlus16_ds8 的生成结果
'''
from utils.utils import load_json_by_line, load_json, save_json
from openai import OpenAI
import os
from tqdm import tqdm

def export_api_key(port):
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:{}'.format(port)
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:{}'.format(port)
    os.environ['FTP_PROXY'] = 'http://127.0.0.1:{}'.format(port)
    os.environ['ALL_PROXY'] = 'http://127.0.0.1:{}'.format(port)
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    os.environ['OPENAI_API_KEY'] = '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'

export_api_key(9999)

class Eval(object):
    def __init__(self):
        self.test_data = load_json_by_line('./data/empathetic_dialogues/data_3.0/test.jsonl')
        self.client = OpenAI()
        self.error_count = 0
        self.bad_count = 0
        self.good_count = 0
        self.not_find_count = 0
        self.generate_results_path = './interact_output/t5_large_rlus16_ds8_generate_results.json'
        self.generate_results_path_2 = './interact_output/t5_large_us10_ds8_generate_results.json'

#     def OpenAI_API_eval(self, dial):
#         emotion = dial['emotion']
#         situation = dial['situation']
#
#         log = dial['log']
#
#         origin_dial = ""
#         tmp_count = 0
#         conv_idx = (int)(dial['dial_id'].split('_')[1])
#         for uttr in self.test_data:
#             if uttr['conv_idx'] == conv_idx:
#                 if tmp_count & 1:
#                     origin_dial += ('Listener: ' + uttr['utterance'] + '\n')
#                 else:
#                     origin_dial += ('Speaker: ' + uttr['utterance'] + '\n')
#                 tmp_count += 1
#
#         generate_dial = f'''Speaker: {log[0]['user']}
# Listener: {log[0]['sys']}
# Speaker: {log[1]['user']}
# Listener: {log[1]['sys']}
# Speaker: {log[2]['user']}
# Listener: {log[2]['sys']}
# Speaker: {log[3]['user']}
# Listener: {log[3]['sys']}
# Speaker: {log[4]['user']}
# Listener: {log[4]['sys']}
# '''
#
#         prompt = f'''[Task Description] I will give you two dialogues. The two sides of the dialogues are Listener and Speaker, respectively.\
# Speaker feel {emotion} because {situation}. \
# Speaker shared these emotions with Listener in dialog, expecting empathy and understanding from them. \
# The two dialogues are as follows:
#
# [First Dialogue]
# {origin_dial}
#
# [Second Dialogue]
# {generate_dial}
#
# [Question]
# In such open-ended dialogs, good listeners demonstrate coherence and maintain a good conversation flow, they display a \
# likeable personality and understanding ofthe speaker. On the contrary, bad listeners don't follow the context and don't \
# show much interest in the conversation. Please choose the one you think is better from the two dialogues, \
# choosing from [First Dialogue] and [Second Dialogue] options.
# '''
#         # print(prompt)
#         # exit()
#
#         try:
#             completion = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#         except Exception as e:
#             print('Exception: {}'.format(e))
#             self.error_count += 1
#             return 0.5
#
#         response = completion.choices[0].message.content
#         if response.find("[First Dialogue]") != -1 or response.find("first dialogue") != -1:
#             score = 0
#             self.bad_count += 1
#         elif response.find("[Second Dialogue]") != -1 or response.find("second dialogue") != -1:
#             score = 1
#             self.good_count += 1
#         else:
#             score = 0.5
#             self.not_find_count += 1
#
#         return score
#
#     def eval_generate_origin(self):
#         cnt = 0
#         generate_data = load_json(self.generate_results_path)
#         for dial in generate_data:
#             cnt += 1
#             dial_id = None
#             for key in dial.keys():
#                 if key != "terminate_reason" and key != "log":  # if key.endswith('json'):
#                     dial_id = key
#             dial['dial_id'] = dial_id
#             goal = dial[dial_id]['goal'].split()
#             dial['emotion'] = goal[1]
#             dial['situation'] = ' '.join(goal[3:])
#
#             score = self.OpenAI_API_eval(dial)
#             dial['gpt-3.5-turbo_score'] = score
#
#             if cnt >= 10:
#                 break
#         save_json(generate_data, './interact_output/t5_large_rlus16_ds8_generate_results_withscore.json')

    def OpenAI_API_eval(self, dial1, dial2):
        emotion = dial1['emotion']
        situation = dial1['situation']

        log1 = dial1['log']
        log2 = dial2['log']

        origin_dial = f'''Speaker: {log1[0]['user']}
Listener: {log1[0]['sys']}
Speaker: {log1[1]['user']}
Listener: {log1[1]['sys']}
Speaker: {log1[2]['user']}
Listener: {log1[2]['sys']}
Speaker: {log1[3]['user']}
Listener: {log1[3]['sys']}
Speaker: {log1[4]['user']}
Listener: {log1[4]['sys']}
'''


        generate_dial = f'''Speaker: {log2[0]['user']}
Listener: {log2[0]['sys']}
Speaker: {log2[1]['user']}
Listener: {log2[1]['sys']}
Speaker: {log2[2]['user']}
Listener: {log2[2]['sys']}
Speaker: {log2[3]['user']}
Listener: {log2[3]['sys']}
Speaker: {log2[4]['user']}
Listener: {log2[4]['sys']}
'''

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
        print(prompt)
        # exit()

        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        except Exception as e:
            print('Exception: {}'.format(e))
            self.error_count += 1
            return 0.5

        response = completion.choices[0].message.content
        # print(response)
        if response.find("[First Dialogue]") != -1 or response.find("first dialogue") != -1:
            score = 0
            self.bad_count += 1
        elif response.find("[Second Dialogue]") != -1 or response.find("second dialogue") != -1:
            score = 1
            self.good_count += 1
        else:
            score = 0.5
            self.not_find_count += 1

        return score

    def eval_generate_origin(self):
        generate_data1 = load_json(self.generate_results_path)
        generate_data2 = load_json(self.generate_results_path_2)

        for dial in generate_data1:
            dial_id = None
            for key in dial.keys():
                if key != "terminate_reason" and key != "log":  # if key.endswith('json'):
                    dial_id = key
            dial['dial_id'] = dial_id
            goal = dial[dial_id]['goal'].split()
            dial['emotion'] = goal[1]
            dial['situation'] = ' '.join(goal[3:])

            # score = self.OpenAI_API_eval(dial)
            # dial['gpt-3.5-turbo_score'] = score
        for dial in generate_data2:
            dial_id = None
            for key in dial.keys():
                if key != "terminate_reason" and key != "log":  # if key.endswith('json'):
                    dial_id = key
            dial['dial_id'] = dial_id
            goal = dial[dial_id]['goal'].split()
            dial['emotion'] = goal[1]
            dial['situation'] = ' '.join(goal[3:])

        new_data = []
        cnt = 0
        for dial1, dial2 in tqdm(zip(generate_data1, generate_data2), desc="PKing", total=len(generate_data1)):
            new_dial = {}
            score = self.OpenAI_API_eval(dial1, dial2)
            # score = 0, 表示 dial1 好
            new_dial['dial_id'] = dial1['dial_id']
            new_dial['emotion'] = dial1['emotion']
            new_dial['situation'] = dial1['situation']
            new_dial['model1'] = "t5_large_rlus16_ds8"
            new_dial['model2'] = "t5_large_us10_ds8"
            new_dial['log1'] = dial1['log']
            new_dial['log2'] = dial2['log']
            new_dial['gpt-3.5-turbo_score'] = score
            new_dial['score_means'] = "score=0, model1 is better, score=1, model2 is better"

            new_data.append(new_dial)
            cnt += 1
            if cnt >= 20:
                break

        # save_json(new_data, './interact_output/pk_result/t5_large_rlus16_ds8_vs_t5_large_us10_ds8.json')

        # print('Good_count = {}, Bad_count = {}, Error_count = {}, Not_Find_count = {}'.format(
        #     self.good_count, self.bad_count, self.error_count, self.not_find_count))

        print('Good_count = {}, Bad_count = {}, Error_count = {}, Not_Find_count = {}'.format(
            self.bad_count, self.good_count, self.error_count, self.not_find_count))




eval = Eval()
eval.eval_generate_origin()
