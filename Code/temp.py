# # #convert test set data to online format
# # from utils.utils import load_json, save_json
# #
# # file1 = 'dialog_t5_small_offline_ep11_onlineformat.json'
# # file2 = 'test_data.json'
# # data1 = load_json(file1)
# # data2 = load_json(file2)
# #
# # for dial in data1:
# #     for key in dial.keys():
# #         if key.endswith('.json'):
# #             dial_id = key
# #             break
# #
# #     for t, turn in enumerate(dial['log']):
# #         turn['sys'] = data2[dial_id]['log'][t]['resp']
# #         turn["belief_states"] = '<bos_belief> ' + data2[dial_id]['log'][t]['constraint'] + ' <eos_belief>'
# #
# # save_json(data1, 'test_data_online_format.json')
#
# from bert_score import BERTScorer
# from utils.utils import load_json, load_json_by_line
#
# def calculate_bert_score(file1, file2):
#     modify = load_json_by_line(file1)  # train
#     origin = load_json(file2)  # tmp
#
#     origins = []
#     for dial in origin:
#         cnt = 0
#         new_dial = {}
#         new_dial['dialog_id'] = dial['dialog_id']
#         for turn in dial['log']:
#             new_dial['utter_idx'] = cnt
#             cnt += 1
#             new_dial['utterance'] = turn['user']
#             origins.append(new_dial.copy())
#
#             new_dial['utter_idx'] = cnt
#             cnt += 1
#             new_dial['utterance'] = turn['sys']
#             origins.append(new_dial.copy())
#
#     references = []
#     candidates = []
#     for i in origins:
#         for j in modify:
#             if i['dialog_id'] == j['conv_idx'] and i['utter_idx'] == j['utter_idx']:
#                 references.append(i['utterance'])
#                 candidates.append(j['utterance'])
#     # print(len(candidates))
#     # print(len(references))
#     assert len(candidates) == len(references)
#
#     # scorer = BERTScorer(lang="en", rescale_with_baseline=True)
#     scorer = BERTScorer(model_type="roberta-large")
#
#     P, R, F1 = scorer.score(candidates, references, verbose=True)
#     print(f"System level F1 score: {F1.mean():.3f}")
#
#     return F1.mean()
#
# ans = 0
# for idx in range(10):
#     file1 = "./interact_output/t5_large_generate_modify/train/train_%03d.json" % idx
#     file2 = "./interact_output/t5_large_generate_modify/tmp/tmp_output_%03d.json" % idx
#     ans += calculate_bert_score(file1, file2)
# print(f"System level F1 score: {ans / 10:.3f}")

# from transformers import T5ForConditionalGeneration
#
# model = T5ForConditionalGeneration.from_pretrained('roberta-large')


'''
用 pytorch 求最大值：
1. 将目标函数取负
2. 将最后的结果（loss）取负

'''

import torch

# 定义目标函数，这里使用一个简单的二次函数
def target_function(x):
    return -x**2 - 5  # 注意取负号，因为我们要进行梯度上升

# 初始化参数，这里假设我们要找到使目标函数最大化的参数值
x = torch.tensor([1.0], requires_grad=True)

# 定义优化器，使用梯度上升，所以lr为正值
optimizer = torch.optim.SGD([x], lr=0.01)

# 迭代优化过程
for epoch in range(1000):
    # 计算目标函数的值
    output = target_function(x)
    output = -output
    # 清零梯度
    optimizer.zero_grad()

    # 计算梯度
    output.backward()

    # 更新参数
    optimizer.step()

# 打印最终的参数值和目标函数值
print("Final parameter value: %.2f" % x.item())
print("Final target function value: %.2f" % target_function(x).item())


