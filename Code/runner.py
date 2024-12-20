import os
import re
import copy
import math
import time
import glob
import shutil
from abc import *
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, get_constant_schedule  # AdamW
from torch.optim import AdamW
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5ForConditionalGeneration
from tensorboardX import SummaryWriter

from reader import MultiWOZIterator, MultiWOZReader, EDIterator, EDReader
from evaluator import MultiWozEvaluator

from utils import definitions
from utils.utils import get_or_create_logger, load_json, save_json, split_user_act_and_resp, split_user_resp, split_system_resp, calculate_bleu


logger = get_or_create_logger(__name__)


class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.belief_loss = 0.0
        self.resp_loss = 0.0

        self.belief_correct = 0.0
        self.resp_correct = 0.0

        self.belief_count = 0.0
        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.resp_loss += step_outputs["resp"]["loss"]
        self.resp_correct += step_outputs["resp"]["correct"]
        self.resp_count += step_outputs["resp"]["count"]

        if 'belief' in step_outputs:
            self.belief_loss += step_outputs["belief"]["loss"]
            self.belief_correct += step_outputs["belief"]["correct"]
            self.belief_count += step_outputs["belief"]["count"]
            do_belief_stats = True
        else:
            do_belief_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_belief_stats)


    def info_stats(self, data_type, global_step, do_belief_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        resp_ppl = math.exp(self.resp_loss / self.resp_count)
        resp_acc = (self.resp_correct / self.resp_count) * 100

        # <MOD>
        if data_type == 'dev':
            self.resp_loss = self.resp_loss / self.global_step * 100  # 10 means log_frequency, 为了和train的loss单位一致

        self.summary_writer.add_scalar(
            "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.resp_loss, resp_ppl, resp_acc)

        if do_belief_stats:
            belief_ppl = math.exp(self.belief_loss / self.belief_count)
            belief_acc = (self.belief_correct / self.belief_count) * 100

            self.summary_writer.add_scalar(
                "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)
            belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
                self.belief_loss, belief_ppl, belief_acc)
        else:
            belief_info = ''

        logger.info(
            " ".join([common_info, resp_info, belief_info,]))

        self.init_stats()

    def info_bleu(self, data_type, bleu, global_step):
        self.summary_writer.add_scalar(
            "{}/bleu".format(data_type), bleu, global_step=global_step)
        logger.info(
            " ".join(["[Validation]", "BLEU-4 Score: {:.2f}".format(bleu)]))


class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader

        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
        else:
            model_path = self.cfg.backbone

        if self.cfg.backbone in ["t5-small", "t5-base", "t5-large"]:
            model = T5ForConditionalGeneration.from_pretrained(model_path)

        logger.info("Load model from {}".format(model_path))

        model.resize_token_embeddings(self.reader.vocab_size)
        model.to(self.cfg.device)
        return model

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


class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        reader = MultiWOZReader(cfg, cfg.version)

        self.iterator = MultiWOZIterator(reader)

        super(MultiWOZRunner, self).__init__(cfg, reader)

    def step_fn(self, inputs, resp_labels, belief_labels=None):
        inputs = inputs.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)
        if self.cfg.agent_type == 'ds' and belief_labels is not None:
             belief_labels = belief_labels.to(self.cfg.device)

        attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)

        encoder_outputs = None
        if self.cfg.agent_type == 'ds' and belief_labels is not None:
            belief_outputs = self.model(input_ids=inputs,
                                        attention_mask=attention_mask,
                                        labels=belief_labels)

            belief_loss = belief_outputs.loss
            belief_logits = belief_outputs.logits
            belief_pred = torch.argmax(belief_logits, dim=-1)

            encoder_last_hidden_state = belief_outputs.encoder_last_hidden_state
            encoder_hidden_states = belief_outputs.encoder_hidden_states
            encoder_attentions = belief_outputs.encoder_attentions
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state,
                                                hidden_states=encoder_hidden_states,
                                                attentions=encoder_attentions)

            # batch_size, max_length = resp_labels.shape[0], resp_labels.shape[1]
            # decoder_attention_mask = torch.ones(batch_size, max_length).to(self.cfg.device) # mask pad and db tokens
            # for i in range(4):
            #     decoder_attention_mask[:,i] = 0

            resp_outputs = self.model(attention_mask=attention_mask,
                                        # decoder_attention_mask=decoder_attention_mask,
                                        encoder_outputs=encoder_outputs,
                                        labels=resp_labels)
            resp_loss = resp_outputs.loss
            resp_logits = resp_outputs.logits
            resp_pred = torch.argmax(resp_logits, dim=-1)

            num_belief_correct, num_belief_count = self.count_tokens(belief_pred, belief_labels, pad_id=self.reader.pad_token_id)
            num_resp_correct, num_resp_count = self.count_tokens(resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        elif self.cfg.agent_type == 'us' and belief_labels is None:
            resp_outputs = self.model(input_ids=inputs,
                                        attention_mask=attention_mask,
                                        labels=resp_labels)
            resp_loss = resp_outputs.loss
            resp_logits = resp_outputs.logits
            resp_pred = torch.argmax(resp_logits, dim=-1)
            num_resp_correct, num_resp_count = self.count_tokens(resp_pred, resp_labels, pad_id=self.reader.pad_token_id)
        else:
            raise Exception('Wrong agent type! It should be us or ds.')

        loss = self.cfg.resp_loss_coeff * resp_loss

        if self.cfg.agent_type == 'ds' and belief_labels is not None:
            loss += self.cfg.bspn_loss_coeff * belief_loss

        step_outputs = {}
        step_outputs["resp"] = {"loss": resp_loss.item(),
                                "correct": num_resp_correct.item(),
                                "count": num_resp_count.item()}
        if self.cfg.agent_type == 'ds':
            step_outputs["belief"] = {"loss": belief_loss.item(),
                                    "correct": num_belief_correct.item(),
                                    "count": num_belief_count.item()}

        return loss, step_outputs

    def train_epoch(self, train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter=None):
        self.model.train()
        self.model.zero_grad()

        with tqdm(total=num_training_steps_per_epoch) as pbar:
            for step, batch in enumerate(train_iterator):
                start_time = time.time()

                inputs, resp_labels, belief_labels = batch

                loss, step_outputs = self.step_fn(inputs, resp_labels, belief_labels)

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

        for epoch in range(1, self.cfg.epochs + 1):
            get_iterator_fn = self.iterator.get_data_iterator(self.cfg.agent_type)
            train_iterator = get_iterator_fn(train_batches, self.cfg.ururu, self.cfg.context_size)

            self.train_epoch(train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter)

            logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            self.save_model(epoch)

            if not self.cfg.no_validation:
                self.validation(reporter.global_step)

    def validation(self, global_step):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)

        get_iterator_fn = self.iterator.get_data_iterator(self.cfg.agent_type)

        dev_iterator = get_iterator_fn(
            dev_batches, self.cfg.ururu, self.cfg.context_size)

        reporter = Reporter(1000000, self.cfg.model_dir)

        torch.set_grad_enabled(False)
        for batch in tqdm(dev_iterator, total=num_steps, desc="Validaction"):
            start_time = time.time()

            inputs, resp_labels, belief_labels = batch

            _, step_outputs = self.step_fn(inputs, resp_labels, belief_labels)

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_belief_stats = True if 'belief' in step_outputs else False

        reporter.info_stats("dev", global_step, do_belief_stats)

        torch.set_grad_enabled(True)

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

                #print(self.reader.tokenizer.decode(input_id))
                #print(self.reader.tokenizer.decode(bspn))

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

                #print(flatten_span)

                #input()

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

    def predict(self):
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains)

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in self.iterator.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                    eval_dial_list.extend(dial_ids)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn["user"]), self.cfg.context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

                attention_mask = torch.where(
                    batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                bspn_decoder_input_ids = self.iterator.tensorize([[self.reader.pad_token_id] + [self.reader.tokenizer.convert_tokens_to_ids(definitions.BOS_BELIEF_TOKEN)] for _ in range(batch_encoder_input_ids.shape[0])])
                bspn_decoder_input_ids = bspn_decoder_input_ids.to(self.cfg.device)

                # belief tracking
                with torch.no_grad():
                    belief_outputs = self.model.generate(input_ids=batch_encoder_input_ids,
                                                         attention_mask=attention_mask,
                                                         decoder_input_ids=bspn_decoder_input_ids,
                                                         eos_token_id=self.reader.eos_token_id,
                                                         max_length=200)

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                decoded_belief_outputs = self.finalize_bspn(
                    belief_outputs, domain_history, constraint_dicts)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])

                if self.cfg.task == "e2e":
                    dbpn = []

                    if self.cfg.use_true_dbpn:
                        for turn in turn_batch:
                            dbpn.append(turn["dbpn"])
                    else:
                        for turn in turn_batch:
                            if self.cfg.add_auxiliary_task:
                                bspn_gen = turn["bspn_gen_with_span"]
                            else:
                                bspn_gen = turn["bspn_gen"]

                            bspn_gen = self.reader.tokenizer.decode(
                                bspn_gen, clean_up_tokenization_spaces=False)

                            db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                      turn["turn_domain"])

                            dbpn_gen = self.reader.encode_text(
                                db_token,
                                bos_token=definitions.BOS_DB_TOKEN,
                                eos_token=definitions.EOS_DB_TOKEN)

                            turn["dbpn_gen"] = dbpn_gen

                            dbpn.append(dbpn_gen)

                    for t, db in enumerate(dbpn):
                        if self.cfg.use_true_curr_aspn:
                            db += turn_batch[t]["aspn"]

                        # T5 use pad_token as start_decoder_token_id
                        dbpn[t] = [self.reader.pad_token_id] + db

                    # aspn has different length
                    if self.cfg.use_true_curr_aspn:
                        for t, _dbpn in enumerate(dbpn):
                            resp_decoder_input_ids = self.iterator.tensorize([_dbpn])

                            resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                            with torch.no_grad():
                                resp_outputs = self.model.generate(
                                    # encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask[t].unsqueeze(0),
                                    decoder_input_ids=resp_decoder_input_ids,
                                    eos_token_id=self.reader.eos_token_id,
                                    max_length=300,)

                                resp_outputs = resp_outputs.cpu().numpy().tolist()

                                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                                turn_batch[t].update(**decoded_resp_outputs[0])

                    else:
                        resp_decoder_input_ids = self.iterator.tensorize(dbpn)

                        resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                        # response generation
                        with torch.no_grad():
                            resp_outputs = self.model.generate(
                                input_ids=batch_encoder_input_ids,
                                # encoder_outputs=encoder_outputs,
                                attention_mask=attention_mask,
                                decoder_input_ids=resp_decoder_input_ids,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=300)

                        resp_outputs = resp_outputs.cpu().numpy().tolist()

                        decoded_resp_outputs = self.finalize_resp(resp_outputs)

                        for t, turn in enumerate(turn_batch):
                            turn.update(**decoded_resp_outputs[t])

                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        if self.cfg.add_auxiliary_task:
                            pv_bspn = turn["bspn_gen_with_span"]
                        else:
                            pv_bspn = turn["bspn_gen"]

                    if self.cfg.use_true_dbpn:
                        pv_dbpn = turn["dbpn"]
                    else:
                        pv_dbpn = turn["dbpn_gen"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    # if self.cfg.use_true_prev_resp:
                    #     if self.cfg.task == "e2e":
                    #         pv_resp = turn["redx"]
                    #     else:
                    #         pv_resp = turn["resp"]
                    # else:
                    #     pv_resp = turn["resp_gen"]

                    if self.cfg.use_true_prev_resp:
                        pv_resp = turn["redx"]
                    else:
                        pv_resp = turn["resp_gen"]

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        # if self.cfg.output:
        #     save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)

        if self.cfg.task == "e2e":
            bleu, success, match = evaluator.e2e_eval(
                results, eval_dial_list=eval_dial_list, add_auxiliary_task=self.cfg.add_auxiliary_task)

            score = 0.5 * (success + match) + bleu

            logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                match, success, bleu, score))

            if self.cfg.output:
                save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))
        else:
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results, add_auxiliary_task=self.cfg.add_auxiliary_task)

            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))

    def us_predict(self):
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains)

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in self.iterator.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                    eval_dial_list.extend(dial_ids)
        
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
                    user_act, user_utterance, _, _ = split_user_act_and_resp(self.reader.tokenizer, model_outputs[t])
                    user_act = self.reader.tokenizer.decode(user_act, clean_up_tokenization_spaces=False).split()
                    user_utterance = self.reader.tokenizer.decode(user_utterance, clean_up_tokenization_spaces=False).split()
                    user_act = ' '.join(user_act[1:-1])
                    user_utterance = ' '.join(user_utterance[1:-1])
                    turn['user_gen'] = user_utterance
                    turn['user_act_gen'] = user_act
                
                    pv_text = copy.copy(turn['user'])
                    pv_text = pv_text + turn['redx']
                    dial_history[t].append(pv_text)
            
            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)
        bleu = evaluator.e2e_eval(results, eval_for_us=True)
        logger.info('bleu: {:2.2f}'.format(bleu))


class EDRunner(BaseRunner):
    def __init__(self, cfg):
        reader = EDReader(cfg)  # MultiWOZReader(cfg, cfg.version)

        self.iterator = EDIterator(reader)  # MultiWOZIterator(reader)

        super(EDRunner, self).__init__(cfg, reader)

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
        修改：增加数据增强
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
            resp_outputs = self.model(input_ids=inputs,  # inputs_embeds=inputs_embeds, <MOD>
                                      attention_mask=attention_mask,
                                      labels=resp_labels)
            resp_loss = resp_outputs.loss
            resp_logits = resp_outputs.logits
            resp_pred = torch.argmax(resp_logits, dim=-1)
            num_resp_correct, num_resp_count = self.count_tokens(resp_pred, resp_labels,
                                                                 pad_id=self.reader.pad_token_id)
        elif self.cfg.agent_type == 'us':
            resp_outputs = self.model(input_ids=inputs,  # inputs_embeds=inputs_embeds,   <MOD>
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

    def train_epoch(self, train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter=None):
        self.model.train()
        self.model.zero_grad()

        with tqdm(total=num_training_steps_per_epoch) as pbar:
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
        bestbleu4, bestloss = 0, 1e8

        for epoch in range(1, self.cfg.epochs + 1):  # <MOD>
            get_iterator_fn = self.iterator.get_data_iterator(self.cfg.agent_type)
            train_iterator = get_iterator_fn(train_batches, self.cfg.ururu, self.cfg.context_size)

            self.train_epoch(train_iterator, optimizer, scheduler, num_training_steps_per_epoch, reporter)

            logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            if not self.cfg.no_validation:
                loss, bleu4 = self.validation(reporter.global_step, num_dialogs=-1)  # self.validation(reporter.global_step) 200指的是测试bleu指标的dev数据量
                # loss, bleu = self.validation(reporter.global_step, epoch)
                # if loss < bestloss:
                if bestbleu4 < bleu4:
                    bestbleu4 = bleu4
                    self.save_model("bestbleu4_ckpt")
                if bestloss > loss:
                    bestloss = loss
                    self.save_model("bestloss_ckpt")
                if epoch == self.cfg.epochs:
                    self.save_model("latest_ckpt")

            logger.info(" ".join(["[Validation]", "Best BLEU-4 Score: {:.2f}  Best Loss: {:.2f}".format(bestbleu4, bestloss)]))
            # self.save_model(epoch)  # <MOD>

    def validation(self, global_step, num_dialogs=-1):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)

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


    def us_predict(self, num_dialogs=-1):  # <MOD>
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
