'''
计算 BLEU，Rouge，Distinct 指标得分
'''

# BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils.utils import load_json

def calculate_bleu(file_path, agent_type):
    results = load_json(file_path)

    score_bleu1 = 0
    score_bleu2 = 0
    score_bleu3 = 0
    score_bleu4 = 0
    count_turns = 0
    smoother = SmoothingFunction()
    for id, dial in results.items():
        for turn in dial:
            count_turns += 1
            if agent_type == 'us':
                references = [turn['user'].split()[1:-1]]
                candidate = turn['user_gen'].split()
            elif agent_type == 'ds':
                references = [turn['resp'].split()[1:-1]]
                candidate = turn['sys_gen'].split()
            else:
                raise Exception

            score_bleu1 += sentence_bleu(references, candidate, weights=(1,), smoothing_function=smoother.method0)
            score_bleu2 += sentence_bleu(references, candidate, weights=(0.5, 0.5), smoothing_function=smoother.method0)
            score_bleu3 += sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33), smoothing_function=smoother.method0)
            score_bleu4 += sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method0)

    return score_bleu1 / count_turns * 100, score_bleu2 / count_turns * 100, score_bleu3 / count_turns * 100, score_bleu4 / count_turns * 100


# Rouge
from rouge import Rouge

def calculate_rouge(file_path, agent_type):
    rouge = Rouge()
    results = load_json(file_path)

    rouge_1_f1 = 0
    rouge_2_f1 = 0
    rouge_l_f1 = 0
    count_turns = 0
    for id, dial in results.items():
        for turn in dial:
            count_turns += 1
            if agent_type == 'us':
                reference = " ".join(turn['user'].split()[1:-1])
                candidate = turn['user_gen']
            elif agent_type == 'ds':
                reference = " ".join(turn['resp'].split()[1:-1])
                candidate = turn['sys_gen']
            else:
                raise Exception

            if candidate == None or candidate == "":
                continue

            rouge_1_f1 += rouge.get_scores(candidate, reference)[0]['rouge-1']['f']
            rouge_2_f1 += rouge.get_scores(candidate, reference)[0]['rouge-2']['f']
            rouge_l_f1 += rouge.get_scores(candidate, reference)[0]['rouge-l']['f']

    return rouge_1_f1 / count_turns * 100, rouge_2_f1 / count_turns * 100, rouge_l_f1 / count_turns * 100


# Distinct
from nltk import ngrams

def calculate_ngram_distinct(file_path, agent_type):
    results = load_json(file_path)

    texts = []
    for id, dial in results.items():
        for turn in dial:
            if agent_type == 'us':
                reference = " ".join(turn['user'].split()[1:-1])
                candidate = turn['user_gen']
            elif agent_type == 'ds':
                reference = " ".join(turn['resp'].split()[1:-1])
                candidate = turn['sys_gen']
            else:
                raise Exception
            text = candidate + reference
            if text == None or text == "" or len(text.split()) < 4:
                continue
            texts.append(text)

    dist_1 = 0
    dist_2 = 0
    dist_3 = 0
    dist_4 = 0
    for text in texts:
        tokens = text.split()
        ngrams_1_list = list(ngrams(tokens, 1))
        ngrams_2_list = list(ngrams(tokens, 2))
        ngrams_3_list = list(ngrams(tokens, 3))
        ngrams_4_list = list(ngrams(tokens, 4))
        dist_1 += len(set(ngrams_1_list)) / len(ngrams_1_list)
        dist_2 += len(set(ngrams_2_list)) / len(ngrams_2_list)
        dist_3 += len(set(ngrams_3_list)) / len(ngrams_3_list)
        dist_4 += len(set(ngrams_4_list)) / len(ngrams_4_list)

    return dist_1 / len(texts) * 100, dist_2 / len(texts) * 100, dist_3 / len(texts) * 100, dist_4 / len(texts) * 100


if __name__ == '__main__':
    agent_type = 'us'
    file_path = "./simulator/simulator_t5_large_data3.0/simulator_rl_dc0.99_lr1e-05_gc1_epoch_4/inference.json"

    bleu1, bleu2, bleu3, bleu4 = calculate_bleu(file_path, agent_type)
    print("BLEU-1:{}; BLEU-2:{}; BLEU-3:{}; BLEU-4:{}".format(bleu1, bleu2, bleu3, bleu4))

    rouge_1_f1, rouge_2_f1, rouge_l_f1 = calculate_rouge(file_path, agent_type)
    print("Rouge-1:{}; Rouge-2:{}; Rouge-L:{}".format(rouge_1_f1, rouge_2_f1, rouge_l_f1))

    dist_1, dist_2, dist_3, dist_4 = calculate_ngram_distinct(file_path, agent_type)
    print("Dist-1:{}; Dist-2:{}; Dist-3:{}; Dist-4:{}".format(dist_1, dist_2, dist_3, dist_4))
