import os
import math
import argparse
import logging
from types import SimpleNamespace
from collections import Counter, OrderedDict
from nltk.util import ngrams
from config import CONFIGURATION_FILE_NAME
from reader import MultiWOZReader
from utils import definitions
from utils.utils import get_or_create_logger, load_json
from utils.clean_dataset import clean_slot_values
from mwzeval.metrics import Evaluator

logger = get_or_create_logger(__name__)


class BLEUScorer:
    """
    BLEU score calculator via GentScorer interface
    it calculates the BLEU-4 by taking the entire corpus in
    Calulate based multiple candidates against multiple references
    """
    def __init__(self):
        pass

    def score(self, parallel_corpus):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        single_turn_bleu = []

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]

            count_single = [0, 0, 0, 0]
            clip_count_single = [0, 0, 0, 0]
            r_single = 0
            c_single = 0
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt
                    count_single[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(
                                max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng]))
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())
                    clip_count_single[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

                r_single += bestmatch[1]
                c_single += len(hyp)

            if c_single == 0:
                single_turn_bleu.append(0)
                continue

            # 计算单轮BLEU
            p0 = 1e-7
            bp = 1 if c_single > r_single else math.exp(1 - float(r_single) / float(c_single))
            p_ns = [float(clip_count_single[i]) / float(count_single[i] + p0) + p0 \
            for i in range(4)]
            s = math.fsum(w * math.log(p_n) \
                    for w, p_n in zip(weights, p_ns) if p_n)
            bleu = bp * math.exp(s)
            single_turn_bleu.append(bleu * 100)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0
                for i in range(4)]
        s = math.fsum(w * math.log(p_n)
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        # return bleu * 100
        return bleu * 100, single_turn_bleu

class MultiWozEvaluator(object):
    def __init__(self, reader, eval_data_type="test"):
        self.reader = reader
        self.all_domains = definitions.ALL_DOMAINS

        self.gold_data = load_json(os.path.join(
            self.reader.data_dir, "{}_data.json".format(eval_data_type)))

        self.eval_data_type = eval_data_type

        self.bleu_scorer = BLEUScorer()

        self.all_info_slot = []
        for d, s_list in definitions.INFORMABLE_SLOTS.items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)

        # only evaluate these slots for dialog success
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']

    def bleu_metric_us(self, data, eval_dial_list=None):
        gen, truth = [], []
        for dial_id, dial in data.items():
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            for turn in dial:
                gen.append(" ".join(turn['user_gen'].split()))
                truth.append(" ".join(turn['user'].split()[1:-1]))
        
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc
    
    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [], []
        for dial_id, dial in data.items():
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            for turn in dial:
                # excepoch <bos_resp>, <eos_resp>
                gen.append(" ".join(turn['resp_gen'].split()[1:-1]))
                truth.append(" ".join(turn['redx'].split()[1:-1]))

        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            # sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
            sc, single_turn_bleu = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
            turn_index = 0
            for dial_id, dial in data.items():
                if eval_dial_list and dial_id not in eval_dial_list:
                    continue
                for turn in dial:
                    turn['resp_BLEU'] = single_turn_bleu[turn_index]
                    turn_index += 1
            print(turn_index, len(single_turn_bleu))
            assert turn_index == len(single_turn_bleu)
        else:
            sc = 0.0
        return sc

    def value_similar(self, a, b):
        return True if a == b else False

        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True
        return False

    def _bspn_to_dict(self, bspn, no_name=False, no_book=False):
        constraint_dict = self.reader.bspn_to_constraint_dict(bspn)

        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s, v in cons.items():
                key = domain+'-'+s
                if no_name and s == 'name':
                    continue
                if no_book:
                    if s in ['people', 'stay'] or \
                       key in ['hotel-day', 'restaurant-day', 'restaurant-time']:
                        continue
                constraint_dict_flat[key] = v

        return constraint_dict_flat

    def _constraint_compare(self, truth_cons, gen_cons,
                            slot_appear_num=None, slot_correct_num=None):
        tp, fp, fn = 0, 0, 0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            # v_truth = truth_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(
                        slot) else slot_correct_num.get(slot)+1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(
                    slot) else slot_appear_num.get(slot)+1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp, fp, fn, acc, list(set(false_slot))

    def dialog_state_tracking_eval(self, dials,
                                   eval_dial_list=None, no_name=False,
                                   no_book=False, add_auxiliary_task=False):
        total_turn, joint_match = 0, 0
        total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0
        slot_appear_num, slot_correct_num = {}, {}
        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            missed_jg_turn_id = []
            for turn_num, turn in enumerate(dial):
                bspn_gen = turn["bspn_gen_with_span"] if add_auxiliary_task else turn["bspn_gen"]

                gen_cons = self._bspn_to_dict(
                    turn['bspn_gen'], no_name=no_name, no_book=no_book)
                truth_cons = self._bspn_to_dict(
                    turn['bspn'], no_name=no_name, no_book=no_book)

                if truth_cons == gen_cons:
                    joint_match += 1
                else:
                    missed_jg_turn_id.append(str(turn['turn_num']))

                if eval_dial_list is None:
                    tp, fp, fn, acc, false_slots = self._constraint_compare(
                        truth_cons, gen_cons, slot_appear_num, slot_correct_num)
                else:
                    tp, fp, fn, acc, false_slots = self._constraint_compare(
                        truth_cons, gen_cons,)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_acc += acc
                total_turn += 1
                # if not no_name and not no_book:
                #     turn['wrong_inform'] = '; '.join(false_slots)   # turn inform metric record

            # dialog inform metric record
            # if not no_name and not no_book:
            #     dial[0]['wrong_inform'] = ' '.join(missed_jg_turn_id)

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        accuracy = total_acc / \
            (total_turn * len(self.all_info_slot) + 1e-10) * 100
        joint_goal = joint_match / (total_turn+1e-10) * 100

        return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num

    def aspn_eval(self, dials, eval_dial_list=None):
        def _get_tp_fp_fn(label_list, pred_list):
            tp = len([t for t in pred_list if t in label_list])
            fp = max(0, len(pred_list) - tp)
            fn = max(0, len(label_list) - tp)
            return tp, fp, fn

        total_tp, total_fp, total_fn = 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_act = []
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                if cfg.same_eval_act_f1_as_hdsa:
                    pred_acts, true_acts = {}, {}
                    for t in turn['aspn_gen']:
                        pred_acts[t] = 1
                    for t in  turn['aspn']:
                        true_acts[t] = 1
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                else:
                    pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
                    true_acts = self.reader.aspan_to_act_list(turn['aspn'])
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                if fp + fn !=0:
                    wrong_act.append(str(turn['turn_num']))
                    turn['wrong_act'] = 'x'

                total_tp += tp
                total_fp += fp
                total_fn += fn

            dial[0]['wrong_act'] = ' '.join(wrong_act)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return f1 * 100

    def context_to_response_eval(self, dials, eval_dial_list=None, add_auxiliary_task=False, add_success_rate=False):
        counts = {}
        for req in self.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}

            for domain in self.all_domains:
                if self.gold_data[dial_id]['goal'].get(domain):
                    true_goal = self.gold_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']

            # print('\n',dial_id)
            success, match, stats, counts = self._evaluateGeneratedDialogue(
                dial, goal, reqs, counts, add_auxiliary_task=add_auxiliary_task)

            if add_success_rate:
                dials[dial_id].append({'success': success, 'inform': match})

            successes += success
            matches += match
            dial_num += 1

            # for domain in gen_stats.keys():
            #     gen_stats[domain][0] += stats[domain][0]
            #     gen_stats[domain][1] += stats[domain][1]
            #     gen_stats[domain][2] += stats[domain][2]

            # if 'SNG' in filename:
            #     for domain in gen_stats.keys():
            #         sng_gen_stats[domain][0] += stats[domain][0]
            #         sng_gen_stats[domain][1] += stats[domain][1]
            #         sng_gen_stats[domain][2] += stats[domain][2]

        # self.logger.info(report)
        succ_rate = successes/(float(dial_num) + 1e-10) * 100
        match_rate = matches/(float(dial_num) + 1e-10) * 100

        return succ_rate, match_rate, counts, dial_num

    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts,
                                   soft_acc=False, add_auxiliary_task=False):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
        #'id'
        requestables = self.requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0:
                continue

            sent_t = turn['resp_gen']
            # sent_t = turn['resp']
            for domain in goal.keys():
                # for computing success
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if add_auxiliary_task:
                            bspn = turn['bspn_gen_with_span']
                        else:
                            bspn = turn['bspn_gen']

                        # bspn = turn['bspn']

                        constraint_dict = self.reader.bspn_to_constraint_dict(
                            bspn)
                        if constraint_dict.get(domain):
                            venues = self.reader.db.queryJsons(
                                domain, constraint_dict[domain], return_name=True)
                        else:
                            venues = []

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            # venue_offered[domain] = random.sample(venues, 1)
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            # flag = False
                            # for ven in venues:
                            #     if venue_offered[domain][0] == ven:
                            #         flag = True
                            #         break
                            # if not flag and venues:
                            flag = False
                            for ven in venues:
                                if ven not in venue_offered[domain]:
                                    # if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if flag and venues:
                            if flag and venues:
                                # sometimes there are no results so sample won't work
                                # print venues
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if '[value_reference]' in sent_t:
                            if 'pointer' not in turn:
                                # In online evaluation, groundtruth booking status is not a available
                                provided_requestables[domain].append('reference')
                            elif 'booked' in turn['pointer'] or 'ok' in turn['pointer']:
                                provided_requestables[domain].append('reference')
                    else:
                        if '[value_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0],
                'hotel': [0, 0, 0],
                'attraction': [0, 0, 0],
                'train': [0, 0, 0],
                'taxi': [0, 0, 0],
                'hospital': [0, 0, 0],
                'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.reader.db.queryJsons(
                    domain, goal[domain]['informable'], return_name=True)
                if type(venue_offered[domain]) is str and \
                   '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and \
                     len(set(venue_offered[domain]) & set(goal_venues))>0:
                    match += 1
                    match_stat = 1
            else:
                if '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request+'_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request+'_offer'] += 1

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                # for request in set(provided_requestables[domain]):
                #     if request in real_requestables[domain]:
                #         domain_success += 1
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1

                # if domain_success >= len(real_requestables[domain]):
                if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats, counts

    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in true_goal[domain]:
                    if 'id' in true_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in true_goal[domain]:
                    for reqs in true_goal[domain]['reqt']:  # addtional requests:
                        if reqs in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(reqs)
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append("reference")

            for s, v in true_goal[domain]['info'].items():
                s_, v_ = clean_slot_values(domain, s, v)
                if len(v_.split()) >1:
                    v_ = ' '.join(
                        [token.text for token in self.reader.nlp(v_)]).strip()
                goal[domain]["informable"][s_] = v_

            if 'book' in true_goal[domain]:
                goal[domain]["booking"] = true_goal[domain]['book']

        return goal

    def run_metrics(self, data, domain="all", file_list=None):
        metric_result = {'domain': domain}

        bleu = self.bleu_metric(data, file_list)

        jg, slot_f1, slot_acc, slot_cnt, slot_corr = self.dialog_state_tracking_eval(
            data, file_list)

        metric_result.update(
            {'joint_goal': jg, 'slot_acc': slot_acc, 'slot_f1': slot_f1})

        info_slots_acc = {}
        for slot in slot_cnt:
            correct = slot_corr.get(slot, 0)
            info_slots_acc[slot] = correct / slot_cnt[slot] * 100
        info_slots_acc = OrderedDict(sorted(info_slots_acc.items(), key=lambda x: x[1]))

        act_f1 = self.aspn_eval(data, file_list)

        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, file_list)

        req_slots_acc = {}
        for req in self.requestables:
            acc = req_offer_counts[req+'_offer']/(req_offer_counts[req+'_total'] + 1e-10)
            req_slots_acc[req] = acc * 100
        req_slots_acc = OrderedDict(sorted(req_slots_acc.items(), key = lambda x: x[1]))

        if dial_num:
            metric_result.update({'act_f1': act_f1,
                'success': success,
                'match': match,
                'bleu': bleu,
                'req_slots_acc': req_slots_acc,
                'info_slots_acc': info_slots_acc,
                'dial_num': dial_num})

            logging.info('[DST] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f',
                         jg, slot_acc, slot_f1, act_f1)
            logging.info('[CTR] match: %2.1f  success: %2.1f  bleu: %2.1f',
                         match, success, bleu)
            logging.info('[CTR] ' + '; '
                         .join(['%s: %2.1f' %(req, acc) for req, acc in req_slots_acc.items()]))

            return metric_result
        else:
            return None

    def e2e_eval(self, data, eval_dial_list=None, add_auxiliary_task=False, eval_for_us=False, online_eval=False, add_success_rate=False):
        if not online_eval:
            if eval_for_us:
                bleu = self.bleu_metric_us(data)
                return bleu
            else:
                bleu = self.bleu_metric(data)
        
        success, match, req_offer_counts, dial_num = self.context_to_response_eval(
            data, eval_dial_list=eval_dial_list, add_success_rate=add_success_rate)

        if online_eval:
            return success, match
        else:
            return bleu, success, match

def convert_results_format(results):
    '''
    修改key：
    sys → resp_gen
    belief_states → bspn_gen

    <MOD> 添加 emotion 和 situation
    '''
    processed_results = {}

    for dial in results:
        dial_id = None
        for key in dial.keys():
            if key != "terminate_reason" and key != "log":  # if key.endswith('json'):
                dial_id = key
        processed_results[dial_id] = []
        for index, single_turn in enumerate(dial['log']):
            new_single_turn = {'turn_num': index}
            if 'sys' in single_turn:
                single_turn['resp_gen'] = single_turn['sys']
                del single_turn['sys']
            if 'belief_states' in single_turn:
                single_turn['bspn_gen'] = single_turn['belief_states']
                del single_turn['belief_states']
            for key, value in single_turn.items():
                new_single_turn[key] = value

            # <MOD>
            goal = dial[dial_id]['goal'].split()
            new_single_turn['emotion'] = goal[1]
            new_single_turn['situation'] = ' '.join(goal[3:])

            processed_results[dial_id].append(new_single_turn)

    return processed_results

def convert_results_format_to_mwzeval(result):
    # 转成官方的评测脚本格式
    def bspn_to_constraint_dict(bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        constraint_dict = OrderedDict()
        domain, slot = None, None
        for token in bspn:
            if token == definitions.EOS_BELIEF_TOKEN:
                break

            if token.startswith("["):
                token = token[1:-1]

                if token in definitions.ALL_DOMAINS:
                    domain = token

                if token.startswith("value_"):
                    if domain is None:
                        continue

                    if domain not in constraint_dict:
                        constraint_dict[domain] = OrderedDict()

                    slot = token.split("_")[1]

                    constraint_dict[domain][slot] = []

            else:
                try:
                    if domain is not None and slot is not None:
                        constraint_dict[domain][slot].append(token)
                except KeyError:
                    continue

        for domain, sv_dict in constraint_dict.items():
            for s, value_tokens in sv_dict.items():
                constraint_dict[domain][s] = " ".join(value_tokens)

        return constraint_dict

    my_predictions = {}
    for dial in result:
        for key in dial.keys():
            if key.endswith('json'):
                dial_id = key[:-5]

        my_predictions[dial_id] = []
        for turn in dial['log']:
            new_turn = {}
            new_turn['response'] = turn['resp_gen']
            new_turn['state'] = bspn_to_constraint_dict(turn['bspn_gen'])
            user_act = turn['user_act'].split()
            if len(user_act) == 0 or user_act[0][1:-1] == 'general':
                turn_domain = ['[general]']
            elif user_act[0][1:-1] not in definitions.ALL_DOMAINS:
                raise Exception('Invalid domain token')
            else:
                turn_domain = [user_act[0][1:-1]]
            new_turn['active_domains'] = turn_domain
            my_predictions[dial_id].append(new_turn)

    return my_predictions

# def convert_results_format_to_mwzeval(result):
#     def bspn_to_constraint_dict(bspn):
#         bspn = bspn.split() if isinstance(bspn, str) else bspn

#         constraint_dict = OrderedDict()
#         domain, slot = None, None
#         for token in bspn:
#             if token == definitions.EOS_BELIEF_TOKEN:
#                 break

#             if token.startswith("["):
#                 token = token[1:-1]

#                 if token in definitions.ALL_DOMAINS:
#                     domain = token

#                 if token.startswith("value_"):
#                     if domain is None:
#                         continue

#                     if domain not in constraint_dict:
#                         constraint_dict[domain] = OrderedDict()

#                     slot = token.split("_")[1]

#                     constraint_dict[domain][slot] = []

#             else:
#                 try:
#                     if domain is not None and slot is not None:
#                         constraint_dict[domain][slot].append(token)
#                 except KeyError:
#                     continue

#         for domain, sv_dict in constraint_dict.items():
#             for s, value_tokens in sv_dict.items():
#                 constraint_dict[domain][s] = " ".join(value_tokens)

#         return constraint_dict

#     my_predictions = {}
#     for dial_id in result:
#         if dial_id.endswith('.json'):
#             dial_id = dial_id[:-5]

#         my_predictions[dial_id] = []
#         for turn in result[dial_id + '.json']:
#             new_turn = {}
#             turn['resp_gen'] = turn['resp_gen'].split()[1:-1]
#             turn['resp_gen'] = ' '.join(turn['resp_gen'])
#             new_turn['response'] = turn['resp_gen']

#             new_turn['state'] = bspn_to_constraint_dict(turn['bspn_gen'])
#             # user_act = turn['user_act'].split()
#             # if len(user_act) == 0 or user_act[0][1:-1] == 'general':
#             #     turn_domain = ['[general]']
#             # elif user_act[0][1:-1] not in definitions.ALL_DOMAINS:
#             #     raise Exception('Invalid domain token')
#             # else:
#             #     turn_domain = [user_act[0][1:-1]]
#             # new_turn['active_domains'] = turn_domain
#             new_turn['active_domains'] = turn['turn_domain']
#             my_predictions[dial_id].append(new_turn)

#     return my_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for evaluation")

    parser.add_argument("-data_type", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("-excluded_domains", type=str, nargs="+")
    parser.add_argument("-eval_type", type=str, default='offline', choices=['offline', 'online'])
    parser.add_argument("-output_result_path", type=str, required=True)
    parser.add_argument("-config_dir", type=str, required=True)

    args = parser.parse_args()

    cfg_path = os.path.join(
        args.config_dir, CONFIGURATION_FILE_NAME)

    cfg = SimpleNamespace(**load_json(cfg_path))

    original_data = load_json(args.output_result_path)
    if args.eval_type == 'online':
        data = convert_results_format(original_data)
        mwzeval_data = convert_results_format_to_mwzeval(original_data)
    else:
        data = original_data

    dial_by_domain = load_json("data/MultiWOZ_2.0/dial_by_domain.json")

    eval_dial_list = None
    if args.excluded_domains is not None:
        eval_dial_list = []
        for domains, dial_ids in dial_by_domain.items():
            domain_list = domains.split("-")

            if len(set(domain_list) & set(args.excluded_domains)) == 0:
                eval_dial_list.extend(dial_ids)

    reader = MultiWOZReader(cfg, cfg.version)

    evaluator = MultiWozEvaluator(reader, args.data_type)
    evaluator2 = Evaluator(bleu=False, success=True, richness=False)

    if cfg.task == "e2e":
        if args.eval_type == 'offline':
            bleu, success, match = evaluator.e2e_eval(
                data, eval_dial_list=eval_dial_list, add_auxiliary_task=cfg.add_auxiliary_task)

            score = 0.5 * (success + match) + bleu

            logger.info('Offline Evaluation: match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f',
                match, success, bleu, score)
        elif args.eval_type == 'online':
            success, match = evaluator.e2e_eval(
                data, eval_dial_list=eval_dial_list, add_auxiliary_task=cfg.add_auxiliary_task, online_eval=True)
            results = evaluator2.evaluate(mwzeval_data)
            logger.info('Online Evaluation V1: match: %2.2f; success: %2.2f;', match, success)
            logger.info('Online Evaluation V2: match: %2.2f; success: %2.2f;', results['success']['inform']['total'], results['success']['success']['total'])
        else:
            raise Exception('Invalid evaluation type.')
    else:
        joint_goal, f1, accuracy, _, _ = evaluator.dialog_state_tracking_eval(
            data, eval_dial_list=eval_dial_list, add_auxiliary_task=cfg.add_auxiliary_task)

        logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;',
            joint_goal, accuracy, f1)
