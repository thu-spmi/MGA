# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# supervised pretraining before RL training
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from eval import MultiWozEvaluator, compute_jacc
from reader import MultiWozReader
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import prepare_for_std_eval
from mwzeval.metrics import Evaluator

import os
import shutil
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import global_config as cfg


#import warnings
#warnings.filterwarnings("ignore")

class Model(object):
    def __init__(self, device=[0]):
        self.device1=device[0]
        self.device2=device[1] if len(device)>1 else device[0]
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        self.reader = MultiWozReader(self.tokenizer)
        self.get_special_ids()
        # logging.info([self.sos_b_id, self.sos_a_id, self.sos_r_id, self.eos_b_id, self.eos_a_id,self.eos_r_id])
        self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if cfg.gradient_checkpoint:
            self.model.config.gradient_checkpointing=True
        self.model.to(self.device1)
            
        self.evaluator = MultiWozEvaluator(self.reader)
        self.std_evaluator=Evaluator(bleu=1, success=1, richness=0)
        if cfg.save_log:
            log_path='./log21/log_{}'.format(cfg.exp_no) if cfg.dataset==1 else './log/log_{}'.format(cfg.exp_no)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        else:
            self.tb_writer = None
        cfg.origin_batch_size=cfg.batch_size
        self.eps=1e-45
        if 'test' not in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        self.global_output=4

    def get_special_ids(self):
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        self.sos_u_id=self.tokenizer.convert_tokens_to_ids('<sos_u>')
        self.eos_u_id=self.tokenizer.convert_tokens_to_ids('<eos_u>')

    def pretrain_session_level(self, posterior=False):
        #logging.info(cfg.mode)
        if cfg.mode=='train':
            num_dials=len(self.reader.train)
            all_batches = self.reader.get_batches('train')
        else:
            #divide the datasets
            if not os.path.exists(cfg.divided_path):
                train_data=self.reader.train
                random.shuffle(train_data)
                bound=int(len(train_data)*int(cfg.spv_proportion)/100)
                self.pre_data=train_data[:bound]
                self.post_data=train_data[bound:]
                encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                logging.info('Divided data saved in %s'%cfg.divided_path)
                json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
            else:
                encoded_data = json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
                self.pre_data=encoded_data['pre_data']
                num_dials=len(self.pre_data)
            all_batches = self.reader.get_batches('train',data=self.pre_data)
            num_dials=len(self.pre_data)

        optimizer, scheduler = self.get_sep_optimizers(num_dials,self.model)

        # log info
        logging.info("***** Running pretraining *****")
        logging.info("  Num Dialogs = %d", num_dials)
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)


        log_inputs = 2
        global_step = 0
        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        if cfg.use_scheduler:
            warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
                if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
            logging.info('warmup epochs:{}'.format(warmup_epochs))
        #eval_loss=self.eval(posterior=posterior,model=self.model)
        #logging.info('initial evaluation loss:%f'%eval_loss)
        num_batch=len(all_batches)
        epoch_th=0.1*cfg.epoch_num if 'distilgpt2' in cfg.gpt_path else -1
        #epoch_th=-1
        for epoch in tqdm(range(cfg.epoch_num)):
            epoch_step = 0
            total_loss=0
            logging_loss=0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            #shuffle batch instead of data
            random.shuffle(all_batches)
            for batch_idx, dial_batch in enumerate(all_batches):
                inputs, labels = self.reader.convert_batch_session(dial_batch,posterior_train=posterior)
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                        logging.info('Input examples:')
                        logging.info(self.tokenizer.decode(inputs['contexts'][0]))
                        log_inputs-=1
                    

                    # to tensor
                    inputs = self.add_torch_input(inputs,posterior=posterior)#B,T
                    labels=self.add_torch_input(labels,posterior=posterior)#B,T
                    
                    # loss
                    outputs = self.model(inputs['contexts_tensor'])
                    if cfg.only_target_loss:
                        loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                    else:
                        loss=self.calculate_loss_and_accuracy(outputs,inputs['contexts_tensor'])
                    if cfg.loss_reg:
                        loss=loss/cfg.gradient_accumulation_steps
                    loss.backward()
                    total_loss+=loss.item()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    epoch_step += 1

                    if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or((batch_idx + 1) == num_batch):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        loss_scalar = (total_loss - logging_loss) / cfg.gradient_accumulation_steps
                        
                        logging_loss = total_loss
                        
                        if self.tb_writer:
                            self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('loss', loss_scalar, global_step)
                            

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(self.model.device):
                                torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, loss: {}'.format(
                epoch, (time.time()-btm)/60, total_loss/epoch_step))
            
            if cfg.evaluate_during_training:
                if cfg.save_type=='min_loss':
                    eval_loss=self.eval(model=self.model,posterior=posterior)
                    logging.info('model evaluation loss:{}'.format(eval_loss))
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    if eval_loss<min_eval_loss:
                        min_eval_loss=eval_loss
                        self.save_model(path='best_loss_model',model=self.model)
                        early_stop_count=cfg.early_stop_count
                    else:
                        if epoch>=warmup_epochs:#early stop after warm up
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
                elif cfg.save_type=='max_score' and epoch>epoch_th:
                    if posterior:
                        eval_result=self.validate_pos(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('act_F1',eval_result['act_F1'],epoch)
                        self.tb_writer.add_scalar('db_acc',eval_result['db_acc'],epoch)
                        score=eval_result['joint_acc']
                    else:
                        eval_result=self.validate_fast(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                        self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                        self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                        self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                        score=eval_result['score']
                    if score>max_score:
                        early_stop_count=cfg.early_stop_count
                        max_score=score
                        self.save_model(path='best_score_model',model=self.model)
                    else:
                        if epoch>=warmup_epochs:
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
            else:#save the model with minimal training loss
                if total_loss/epoch_step<min_loss:
                    min_loss=total_loss/epoch_step
                    self.save_model(posterior=posterior,model=self.model)
    
    def pretrain_turn_level(self, posterior=False):
        if cfg.mode=='train':
            num_dials=len(self.reader.train)
            all_batches = self.reader.get_batches('train')
        else:
            #divide the datasets
            if not os.path.exists(cfg.divided_path):
                train_data=self.reader.train
                temp_path=os.path.join(cfg.data_path,'divided_data{}.json'.format(cfg.spv_proportion-5))
                #logging.info(temp_path)
                if os.path.exists(temp_path):
                    encoded_data = json.loads(open(temp_path, 'r', encoding='utf-8').read())
                    add_len=int(0.05*len(train_data))
                    self.pre_data=encoded_data['pre_data']+encoded_data['post_data'][:add_len]
                    self.post_data=encoded_data['post_data'][add_len:]
                    encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                    logging.info('Divide data from %s, saved in %s'%(temp_path, cfg.divided_path))
                    json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
                else:
                    random.shuffle(train_data)
                    bound=int(len(train_data)*int(cfg.spv_proportion)/100)
                    self.pre_data=train_data[:bound]
                    self.post_data=train_data[bound:]
                    encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                    logging.info('Divided data saved in %s'%cfg.divided_path)
                    json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
            else:
                encoded_data = json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
                logging.info('Reading data from {}'.format(cfg.divided_path))
                self.pre_data=encoded_data['pre_data']
                num_dials=len(self.pre_data)
            all_batches = self.reader.get_batches('train',data=self.pre_data)
            num_dials=len(self.pre_data)
        set_stats = self.reader.set_stats['train']
        num_turns=set_stats['num_turns']
        optimizer, scheduler = self.get_sep_optimizers(num_turns,self.model)

        # log info
        logging.info("***** Running turn-level training *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)

        log_inputs = 6
        global_step = 0

        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        #epoch_th=-1
        epoch_th=0.2*cfg.epoch_num if 'distilgpt2' in cfg.gpt_path else -1
        warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
            if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        c1, c2 = 0,0
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            random.shuffle(all_batches)
            #data_iterator = self.reader.get_data_iterator(all_batches)

            for batch_idx, batch0 in enumerate(all_batches):
                dial_batch=self.reader.transpose_batch(batch0)
                pv_batch = None
                c1+=1
                for turn_num, turn_batch in enumerate(dial_batch):
                    c2+=1
                    first_turn = (turn_num == 0)
                    inputs, labels = self.reader.convert_batch_turn(
                            turn_batch, pv_batch, first_turn, posterior=posterior)
                    pv_batch = self.reader.get_pv_batch(pv_batch, user=turn_batch['user'],
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'], side='sys')
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                            log_inputs-=1
                        inputs = self.add_torch_input(inputs)
                        outputs = self.model(inputs['contexts_tensor'])
                        if cfg.only_target_loss:
                            labels=self.add_torch_input(labels)    
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                        else:
                            loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if epoch_step % cfg.gradient_accumulation_steps == 0 or(
                            batch_idx==len(all_batches)-1 and turn_num==len(dial_batch)-1):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1
                            

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            if epoch==0:
                logging.info('Num dials:{}, num_turns:{}'.format(c1, c2))
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                epoch, (time.time()-btm)/60, tr_loss))
            if cfg.evaluate_during_training:
                if cfg.save_type=='min_loss':
                    eval_loss=self.eval(model=self.model,posterior=posterior)
                    logging.info('model evaluation loss:{}'.format(eval_loss))
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    if eval_loss<min_eval_loss:
                        min_eval_loss=eval_loss
                        self.save_model(path='best_loss_model',model=self.model)
                        early_stop_count=cfg.early_stop_count
                    else:
                        if epoch>=warmup_epochs:#early stop after warm up
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
                elif cfg.save_type=='max_score' and epoch>epoch_th:
                    if posterior:
                        eval_result=self.validate_pos(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('act_F1',eval_result['act_F1'],epoch)
                        self.tb_writer.add_scalar('db_acc',eval_result['db_acc'],epoch)
                        score=eval_result['joint_acc']
                    else:
                        eval_result=self.validate_fast(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                        self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                        self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                        self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                        score=eval_result['score']
                    if score>max_score:
                        early_stop_count=cfg.early_stop_count
                        max_score=score
                        self.save_model(path='best_score_model',model=self.model)
                    else:
                        if epoch>=warmup_epochs:
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
            else:# save the model for every epoch
                self.save_model(path='model_{}'.format(epoch), model=self.model)
    

    def get_sep_optimizers(self,num_dials,model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = num_dials*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.origin_batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps) if cfg.use_scheduler else None
        return optimizer, scheduler

    def add_torch_input(self, inputs, posterior=False):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        if posterior:
            contexts_tensor = contexts_tensor.to(self.device2)
        else:
            contexts_tensor = contexts_tensor.to(self.device1)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs


    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    
    def save_model(self, posterior=False, path=None, model=None):
        if not path:
            if posterior:
                save_path = os.path.join(cfg.exp_path, 'best_model_post')
            else:
                save_path = os.path.join(cfg.exp_path, 'best_model_pri')
        else:
            save_path = os.path.join(cfg.exp_path, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        if not model:
            if posterior:
                self.PosteriorModel.save_pretrained(save_path)
            else:
                self.PrioriModel.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # save cfg


    def eval(self,data='dev', posterior=False, model=None):
        if cfg.turn_level:
            return self.eval_turn_level(data, posterior=posterior, model=model)
        model.eval()
        temp=cfg.batch_size
        cfg.batch_size=cfg.origin_batch_size
        all_batches = self.reader.get_batches(data)
        total_batch=len(all_batches)
        total_loss=0
        with torch.no_grad():
            for batch in all_batches:
                if batch==[]:
                    continue
                inputs,labels=self.reader.convert_batch_session(batch,posterior_train=posterior)
                inputs=self.add_torch_input(inputs)#B,T
                labels=self.add_torch_input(labels)#B,T
                outputs = model(inputs['contexts_tensor'])
                loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                total_loss+=loss.item()
        cfg.batch_size=temp
        return total_loss/total_batch
    
    def eval_turn_level(self,data='dev',posterior=False,model=None):
        model.eval()
        all_batches = self.reader.get_batches(data)
        total_batch=len(all_batches)
        total_loss=0
        with torch.no_grad():
            data_iterator = self.reader.get_data_iterator(all_batches)
            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn, side='sys', posterior=posterior)
                    pv_batch = self.reader.get_pv_batch(pv_batch, user=turn_batch['user'],
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'], side='sys')
                    inputs=self.add_torch_input(inputs)#B,T
                    labels=self.add_torch_input(labels)#B,T
                    outputs = model(inputs['contexts_tensor'])
                    loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                    total_loss+=loss.item()
        return total_loss/total_batch

    def validate_fast(self,data='dev'):
        if cfg.mode=='pretrain' or cfg.mode=='train':
            self.PrioriModel=self.model
            self.device1=self.model.device
        
        self.PrioriModel.eval()
        eval_data = self.reader.get_eval_data(data)
        if cfg.debugging:
            eval_data=eval_data[:32]
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)
        result_path=os.path.join(cfg.eval_load_path,'result.json')
        
        if os.path.exists(result_path) and cfg.mode=='test':
            #results,field=self.reader.load_result(result_path)
            results=json.load(open(result_path, 'r'))
            joint_acc=compute_jacc(results)
            #joint_acc=0
            cfg.use_true_bspn_for_ctr_eval=False
            bleu, success, match = self.evaluator.validation_metric(results)
            score = 0.5 * (success + match) + bleu
            logging.info('[Old] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
            
            input_data=prepare_for_std_eval(data=results)
            std_metrics = self.std_evaluator.evaluate(input_data)
            bleu=std_metrics['bleu']['damd']
            match=std_metrics['success']['inform']['total']
            success=std_metrics['success']['success']['total']
            score = 0.5 * (success + match) + bleu
            logging.info(std_metrics)
            logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))

            eval_results = {}
            eval_results['bleu'] = bleu
            eval_results['success'] = success
            eval_results['match'] = match
            eval_results['score'] = score
            eval_results['joint_acc']=joint_acc
            return eval_results
        
        # valid_losses = []
        result_collection = {}
        st=time.time()
        for batch in batches:
            try:
                if batch==[]:
                    continue
                if cfg.turn_level:
                    batch=self.generate_batch_turn_level(batch)
                else:
                    batch=self.generate_batch_e2e(batch)
                for dialog in batch:
                    result_collection.update(self.reader.inverse_transpose_turn(dialog))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device1):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        results, field = self.reader.wrap_result_lm(result_collection)
        logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))

        joint_acc=compute_jacc(results)
        #joint_acc=0
        cfg.use_true_bspn_for_ctr_eval=False
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        logging.info('[Old] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        input_data=prepare_for_std_eval(data=results)
        std_metrics = self.std_evaluator.evaluate(input_data)
        bleu=std_metrics['bleu']['damd']
        match=std_metrics['success']['inform']['total']
        success=std_metrics['success']['success']['total']
        score = 0.5 * (success + match) + bleu
        #logger = logging.getLogger()
        #logger.setLevel(logging.INFO)
        if cfg.mode=='test':
            logging.info(std_metrics)
        
        logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        json.dump(results, open(result_path, 'w'), indent=2)
        #self.reader.save_result('w', results, field,result_name='result.csv')

        eval_results = {}
        eval_results['bleu'] = std_metrics['bleu']['damd']
        eval_results['success'] = std_metrics['success']['success']['total']
        eval_results['match'] = std_metrics['success']['inform']['total']
        eval_results['score'] = score
        eval_results['joint_acc']=joint_acc
        cfg.batch_size=cfg.origin_batch_size
        return eval_results

    def generate_batch_session_level(self, batch):
        bs_max_len=75
        resp_max_len=80 if cfg.model_act else 60
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        sos_r_id=self.sos_r_id
        eos_a_id=self.eos_a_id
        eos_r_id=self.eos_r_id
        
        batch_size=len(batch)
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        with torch.no_grad():
            for turn_num in range(len(batch[0])):
                past_key_values=None
                end_flag=np.zeros(len(batch))
                contexts=self.reader.convert_eval_batch_session(batch,contexts,turn_num,bs_gen,\
                        prior=True,resp_gen=resp_gen)
                '''
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                '''
                inputs,attentions=self.reader.batch_align(contexts,left_len=bs_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(self.device1)
                attentions=torch.tensor(attentions).to(self.device1)
                if not cfg.use_true_curr_bspn:#generate
                    for i in range(bs_max_len):
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        if past_key_values is not None:
                            position_ids=position_ids[:, -1].unsqueeze(-1)
                        outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)

                        past_key_values=outputs.past_key_values

                        preds=outputs.logits[:,-1,:].argmax(-1)#B
                        if i==0:
                            bs_tensor=preds.unsqueeze(1)
                        else:
                            bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(self.device1)),dim=1)
                        inputs=preds.unsqueeze(1)
                        end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                        if sum(end_flag==0)==0:
                            break
                    bs_gen,db_gen=self.reader.get_bspn(bs_tensor,return_db=True,data=batch,turn_num=turn_num)
                else:
                    for dial in batch:
                        bs_gen.append(dial[turn_num]['bspn'])
                        db_gen.append(dial[turn_num]['aspn'])
                past_key_values=None
                end_flag=np.zeros(len(batch))
                contexts=self.reader.convert_eval_batch_session(batch,contexts,turn_num,bs_gen,\
                    prior=True,db_gen=db_gen)
                '''
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                '''
                inputs,attentions=self.reader.batch_align(contexts,left_len=resp_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(self.device1)#B,T
                attentions=torch.tensor(attentions).to(self.device1)
                for i in range(resp_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)

                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)
                    past_key_values=outputs.past_key_values
                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        resp_tensor=preds.unsqueeze(1)
                    else:
                        resp_tensor=torch.cat([resp_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(self.device1)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_r_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                resp_gen=self.reader.get_resp(resp_tensor)#if cfg.model_act then resp_gen contains act_gen and resp_gen
                for i in range(len(batch)):
                    batch[i][turn_num]['bspn_gen']=bs_gen[i]
                    #batch[i][turn_num]['db']=db_gen[i]
                    if cfg.model_act:
                        temp=resp_gen[i]
                        if eos_a_id in temp:
                            batch[i][turn_num]['aspn_gen']=temp[:temp.index(eos_a_id)+1]
                        else:
                            batch[i][turn_num]['aspn_gen']=temp[:-1]+[eos_a_id]
                        if sos_r_id in temp:
                            batch[i][turn_num]['resp_gen']=temp[temp.index(sos_r_id):]
                        else:
                            batch[i][turn_num]['resp_gen']=[sos_r_id]+temp[1:]
                        aspn_temp = batch[i][turn_num]['aspn'] if cfg.use_true_prev_aspn else batch[i][turn_num]['aspn_gen']
                        resp_temp = batch[i][turn_num]['resp'] if cfg.use_true_prev_resp else batch[i][turn_num]['resp_gen']
                        resp_gen[i] = aspn_temp+resp_temp
                    else:
                        batch[i][turn_num]['resp_gen']=resp_gen[i]
                        resp_gen[i] = batch[i][turn_num]['resp'] if cfg.use_true_prev_resp else batch[i][turn_num]['resp_gen']
        return batch
    
    def generate_batch_turn_level(self, batch):
        
        batch=self.reader.transpose_batch(batch)

        bs_max_len=75
        resp_max_len=80
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        sos_r_id=self.sos_r_id
        eos_a_id=self.eos_a_id
        eos_r_id=self.eos_r_id

        batch_size=len(batch[0]['dial_id'])
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        pv_batch=None

        device=self.device1
        with torch.no_grad():
            for turn_num, turn_batch in enumerate(batch):
                # generate bspn
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn')
                inputs,attentions=self.reader.batch_align(contexts,left_len=bs_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(bs_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        bs_tensor=preds.unsqueeze(1)
                    else:
                        bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                bs_gen,db_gen=self.reader.get_bspn(bs_tensor,return_db=True,data=batch,turn_num=turn_num)
                # generate aspn and resp
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
                    bspn_gen=bs_gen,db_gen=db_gen)
                
                #if self.global_output>0 and cfg.mode=='test':
                 #   logging.info(self.tokenizer.decode(contexts[0]))
                inputs,attentions=self.reader.batch_align(contexts,left_len=resp_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(resp_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)
                    past_key_values=outputs.past_key_values
                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        resp_tensor=preds.unsqueeze(1)
                    else:
                        resp_tensor=torch.cat([resp_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_r_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                resp_gen=self.reader.get_resp(resp_tensor)
                aspn_gen=[]
                for i, temp in enumerate(resp_gen):
                    if eos_a_id in temp:
                        aspn=temp[:temp.index(eos_a_id)+1]
                    else:
                        aspn=temp[:-1]+[eos_a_id]
                    if sos_r_id in temp:
                        resp=temp[temp.index(sos_r_id):]
                    else:
                        resp=[sos_r_id]+temp[1:]
                    resp_gen[i]=resp
                    aspn_gen.append(aspn)
                pv_batch=self.reader.get_pv_batch(pv_batch, user=turn_batch['user'], resp=resp_gen, bspn=bs_gen)
                turn_batch['bspn_gen']=bs_gen
                turn_batch['aspn_gen']=aspn_gen
                turn_batch['resp_gen']=resp_gen
                turn_batch['db_gen']=db_gen
        return self.reader.inverse_transpose_batch(batch)

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return



def main():
    if not os.path.exists('./experiments') and cfg.dataset==0:
        os.mkdir('./experiments')

    if not os.path.exists('./experiments_21') and cfg.dataset==1:
        os.mkdir('./experiments_21')
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    cfg._init_logging_handler(args.mode)
    if 'test' in args.mode:
        parse_arg_cfg(args)
        cfg.eval_load_path=cfg.gpt_path
    else:  # train
        parse_arg_cfg(args)
        if cfg.exp_path in ['', 'to be generated']:
            # generate exp_path
            experiments_path = './experiments_21' if cfg.dataset==1 else './experiments'
            if cfg.exp_no=='':
                if cfg.mode=='pretrain':
                    cfg.exp_no = 'pre_pos' if cfg.posterior_train else 'pre_'
                    cfg.exp_no += str(cfg.spv_proportion)
                elif cfg.mode=='train':
                    cfg.exp_no='full'
                    if cfg.posterior_train:
                        cfg.exp_no += '_pos'
            #print('exp_no:',cfg.exp_no)
            cfg.exp_path = os.path.join(experiments_path, cfg.exp_no)
            if cfg.save_log and not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.eval_load_path = cfg.exp_path
    
    logging.info('Model path:{}'.format(cfg.eval_load_path))
    device=cfg.cuda_device
    cfg.divided_path=os.path.join(cfg.data_path,'divided_data{}.json'.format(cfg.spv_proportion))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Model(device)

    if args.mode =='pretrain' or args.mode=='train':
        if cfg.turn_level:
            m.pretrain_turn_level(posterior=cfg.posterior_train)
        else:
            m.pretrain_session_level(posterior=cfg.posterior_train)
    else:  # test
        m.validate_fast('test')


if __name__ == "__main__":
    main()
