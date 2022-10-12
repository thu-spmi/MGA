# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from eval import MultiWozEvaluator
from reader import MultiWozReader
import torch
import torch.nn as nn
import torch.nn.functional as F
from mwzeval.metrics import Evaluator
import copy

import os
import shutil
import random
import argparse
import time
import logging
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import global_config as cfg
from pretrain import Model

class Semi_Model(Model):
    
    def __init__(self, device=[0]):
        self.device1=device[0]
        self.device2=device[1] if len(device)>1 else device[0]
        logging.info('PrioriModel sets on GPU{}, PosteriorModel sets on GPU{}'.format(self.device1,self.device2))
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.PrioriModel_path)
        self.reader = MultiWozReader(self.tokenizer)
        self.get_special_ids()

        # create model: gpt2
        self.PrioriModel = GPT2LMHeadModel.from_pretrained(cfg.PrioriModel_path)
        self.PosteriorModel=GPT2LMHeadModel.from_pretrained(cfg.PosteriorModel_path)
        logging.info("model loaded from {} and {}".format(cfg.PrioriModel_path,cfg.PosteriorModel_path))
        self.PrioriModel.resize_token_embeddings(len(self.tokenizer))
        self.PosteriorModel.resize_token_embeddings(len(self.tokenizer))
        if cfg.gradient_checkpoint:
            self.PrioriModel.config.gradient_checkpointing=True
            self.PosteriorModel.config.gradient_checkpointing=True
        self.PrioriModel.to(self.device1)
        self.PosteriorModel.to(self.device2)

        self.vocab_size=len(self.tokenizer)
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
    
    
    def semi_VL_turn_level(self):
        logging.info('------Running turn-level variational learning------')
        data=json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
        data_lab=data['pre_data']
        data_unl=data['post_data']
        logging.info('Labeled dials:{}, unlabeled dials:{}'.format(len(data_lab),len(data_unl)))
        num_dials=len(data_lab)+len(data_unl)
        
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        batches_lab=self.reader.get_batches('train',data=data_lab)
        label_turns=self.reader.set_stats['train']['num_turns']
        batches_unl=self.reader.get_batches('train',data=data_unl)
        unlabel_turns=self.reader.set_stats['train']['num_turns']
        all_batches=[]
        data_repeat=3 if cfg.spv_proportion<=15 else 1
        label_turns*=data_repeat
        num_turns=label_turns+unlabel_turns
        for _ in range(data_repeat-1):
            num_dials+=len(data_lab)

        if cfg.debugging:
            batches_lab=[]
            batches_unl=batches_unl[:len(batches_unl)//15]

        for _ in range(data_repeat):
            for batch in batches_lab:
                all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':True})
        for batch in batches_unl:
            all_batches.append({'batch':self.reader.transpose_batch(batch),'supervised':False})
        batch_num=sum([len(item['batch']) for item in all_batches])
        logging.info('Total turns:{}, steps:{}'.format(num_turns, batch_num))
        # cleare memory
        batches_lab=[]
        batches_unl=[]
        optimizer1, scheduler1 = self.get_sep_optimizers(num_turns,self.PrioriModel, num_batches=batch_num)
        optimizer2, scheduler2 = self.get_sep_optimizers(num_turns,self.PosteriorModel, num_batches=batch_num)

        # log info
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))
        log_inputs = 3
        global_step = 0
        max_score=0
        early_stop_count=cfg.early_stop_count
        if cfg.use_scheduler:
            warmup_epochs=cfg.warmup_steps*cfg.batch_size//num_dials if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
            logging.info('Warmup epochs:{}'.format(warmup_epochs))
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss, sup_loss, uns_loss = 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.zero_grad()
            self.PosteriorModel.zero_grad()
            random.shuffle(all_batches)

            for batch_idx, dial_batch_dict in enumerate(all_batches):
                pv_batch=None
                pv_bspn_batch=None
                turn_domain_batch=[[] for _ in range(len(dial_batch_dict['batch'][0]['dial_id']))]
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if dial_batch_dict['supervised']==False: # use the inference model to generate pseudo labels
                        turn_batch, next_pv_batch, pv_bspn_batch, turn_domain_batch=self.gen_turn_hidden_state(turn_batch,
                            pv_batch, posterior=True, pv_bspn_batch=pv_bspn_batch, turn_domain_batch=turn_domain_batch)
                    else:
                        next_pv_batch=self.reader.get_pv_batch(pv_batch, user=turn_batch['user'], 
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=self.reader.split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    for i, batch in enumerate(mini_batches):
                        mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                        if not dial_batch_dict['supervised'] and not cfg.VL_ablation:
                            # if the batch is not unsupervised, then we conduct unsupervised VL method
                            # ablation study: treat all batches, whether labeled or pseudo labeled, as supervised batches
                            inputs_prior, labels_prior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=True)
                            self.PrioriModel.train()
                            self.PosteriorModel.train()
                            if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                                logging.info('Prior examples:\n{}'.format(self.tokenizer.decode(inputs_prior['contexts'][0])))
                                logging.info("Posterior examples:\n{}".format(self.tokenizer.decode(inputs_posterior['contexts'][0])))
                                log_inputs -= 1

                            # to tensor
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                            # loss
                            outputs_prior=self.PrioriModel(inputs_prior['contexts_tensor'])
                            outputs_posterior=self.PosteriorModel(inputs_posterior['contexts_tensor'])#B,T,V
                            logits_pri=outputs_prior[0]
                            logits_post=outputs_posterior[0]
                            #straight through trick
                            ST_inputs_prior, resp_label=self.get_turn_ST_input(inputs_prior['contexts_tensor'],\
                                    logits_post,labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            loss_kl=self.get_turn_kl_loss(logits_pri,logits_post.to(self.device1),\
                                    labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'].to(self.device1))
                            
                            embed_prior=ST_inputs_prior.matmul(self.PrioriModel.get_input_embeddings().weight)#multiple the input embedding
                            outputs1=self.PrioriModel(inputs_embeds=embed_prior)
                            loss_ce=self.calculate_loss_and_accuracy(outputs1, resp_label)
                            loss=loss_ce+cfg.kl_loss_weight*loss_kl
                            if cfg.loss_reg:
                                loss=loss/cfg.gradient_accumulation_steps
                            loss.backward()
                            tr_loss+=loss.item()
                            uns_loss+=loss.item()
                            uns_step+=1
                            torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                            torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                        else:# supervised training
                            inputs_prior, labels_prior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=False)
                            inputs_posterior, labels_posterior = self.reader.convert_batch_turn(batch, mini_pv_batch, first_turn, posterior=True)
                            if log_inputs > 0 and cfg.example_log and not dial_batch_dict['supervised']: 
                                logging.info('Prior examples:\n{}'.format(self.tokenizer.decode(inputs_prior['contexts'][0])))
                                logging.info("Posterior examples:\n{}".format(self.tokenizer.decode(inputs_posterior['contexts'][0])))
                                log_inputs -= 1
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                            outputs1 = self.PrioriModel(inputs_prior['contexts_tensor'])
                            loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            outputs2=self.PosteriorModel(inputs_posterior['contexts_tensor'])
                            loss_pos=self.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])

                            if cfg.loss_reg:
                                loss_pri=loss_pri/cfg.gradient_accumulation_steps
                                loss_pos=loss_pos/cfg.gradient_accumulation_steps
                            loss=loss_pri+loss_pos.to(self.device1)
                            loss.backward()
                            tr_loss+=loss.item()
                            sup_loss+=loss.item()
                            sup_step+=1
                            torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                            torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                    epoch_step+=1
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    global_step+=1
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    if self.tb_writer:
                        self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('loss', loss.item(), global_step)
                    pv_batch=next_pv_batch
            
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/epoch_step, sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10)))
            eval_result=self.validate_fast(data='dev')
            if self.tb_writer:
                self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
            
            if eval_result['score']>max_score:
                max_score=eval_result['score']
                self.save_model(path='best_score_model')
                early_stop_count=cfg.early_stop_count
            else:
                weight_decay_count-=1
                if weight_decay_count==0 and not cfg.use_scheduler:
                    lr=lr*cfg.lr_decay
                    for group in optimizer1.param_groups:
                        group['lr'] = lr
                    for group in optimizer2.param_groups:
                        group['lr'] = lr
                    logging.info("learning rate decay to {}".format(lr))
                    weight_decay_count = cfg.weight_decay_count
                if epoch>=warmup_epochs:
                    early_stop_count-=1
                    logging.info('early stop count:%d'%early_stop_count)
            if lr<1e-9 and not cfg.use_scheduler:
                logging.info('learning rate too small, break')
                break
            if early_stop_count==0 and cfg.early_stop:
                logging.info('early stopped')
                break
    
    def semi_VL_session_level(self):
        logging.info('------Running session-level variational learning------')
        data = json.loads(open(cfg.divided_path,'r', encoding='utf-8').read())
        data_lab=data['pre_data']
        data_unl=data['post_data']
        logging.info('Labeled dials:{}, unlabeled dials:{}'.format(len(data_lab),len(data_unl)))
        num_dials=len(data_lab)+len(data_unl)

        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        batches_lab=self.reader.get_batches('train',data=data_lab)
        all_batches=[]
        data_repeat=3 if cfg.spv_proportion==10 else 1
        for _ in range(data_repeat-1):
            num_dials+=len(data_lab)

        for _ in range(data_repeat):
            for batch in batches_lab:
                all_batches.append({'batch':batch,'supervised':True})

        batches_unl=self.reader.get_batches('train',data=data_unl)
        for batch in batches_unl:
            all_batches.append({'batch':batch,'supervised':False,'dataset':'all'})

        optimizer1, scheduler1 = self.get_sep_optimizers(num_dials,self.PrioriModel)
        optimizer2, scheduler2 = self.get_sep_optimizers(num_dials,self.PosteriorModel)
        logging.info("***** Running training *****")
        logging.info("  Num Dialogs = %d", num_dials)
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     num_dials*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size))


        log_inputs = 2
        global_step = 0
        max_score=0
        epoch_num=cfg.epoch_num
        early_stop_count=cfg.early_stop_count
        warmup_epochs=cfg.warmup_steps*cfg.batch_size//num_dials if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        if cfg.use_scheduler:
            logging.info('warmup epochs:{}'.format(warmup_epochs))
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in range(epoch_num):
            epoch_step = 0
            epoch_step_uns=0
            epoch_step_sup=0
            tr_loss = 0.0
            loss1=0
            loss2=0
            loss_uns=0
            loss_sup=0
            logging_loss = 0.0
            gen_time=0
            forward_time=0 # including backward time 
            backward_time=0
            btm = time.time()

            random.shuffle(all_batches)
            for batch_idx, dial_batch_dict in enumerate(all_batches):
                self.PrioriModel.zero_grad()
                self.PosteriorModel.zero_grad()
                turn_nums=[len(dial) for dial in dial_batch_dict['batch']]
                assert all([turn_num==turn_nums[0] for turn_num in turn_nums])
                if dial_batch_dict['supervised']==False:
                    st1=time.time()
                    dial_batch_large=self.gen_session_hidden_state(dial_batch_dict['batch'],model_name='PosteriorModel')
                    gen_time+=time.time()-st1
                    dial_batch=[]
                    st2=time.time()
                    for i, dial in enumerate(dial_batch_large):
                        dial_batch.append(dial)
                        if len(dial_batch)==cfg.origin_batch_size or i==len(dial_batch_large)-1:
                            try:
                                inputs_prior, labels_prior, bspn_labels_pri = \
                                    self.reader.convert_batch_session(dial_batch,only_resp_label=True,bspn_label=True)
                                inputs_posterior, labels_posterior = \
                                    self.reader.convert_batch_session(dial_batch,posterior_train=True)

                                self.PrioriModel.train()
                                self.PosteriorModel.train()
                                if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                                    logging.info('Prior examples')
                                    logging.info(self.tokenizer.decode(inputs_prior['contexts'][0]))
                                    logging.info("Posterior examples")
                                    logging.info(self.tokenizer.decode(inputs_posterior['contexts'][0]))
                                    log_inputs -= 1

                                # to tensor
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                                bspn_labels_pri=self.add_torch_input(bspn_labels_pri)
                                # loss
                                outputs_posterior=self.PosteriorModel(inputs_posterior['contexts_tensor'])#B,T,V
                                logits_post=outputs_posterior[0]
                                outputs_prior=self.PrioriModel(inputs_prior['contexts_tensor'])                                   
                                logits_pri=outputs_prior[0]
                                #straight through trick
                                ST_inputs_prior=self.get_session_ST_input(inputs_prior['contexts_tensor'],\
                                        logits_post,bspn_labels_pri['contexts_tensor'],labels_posterior['contexts_tensor'])
                                loss_kl=self.get_session_kl_loss(logits_pri,logits_post,\
                                    bspn_labels_pri['contexts_tensor'],labels_posterior['contexts_tensor'])
                                embed_prior=ST_inputs_prior.matmul(self.PrioriModel.get_input_embeddings().weight)#multiple the input embedding
                                outputs1=self.PrioriModel(inputs_embeds=embed_prior)
                                loss_ce=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            
                                loss=loss_ce+cfg.kl_loss_weight*loss_kl
                                if cfg.loss_reg:
                                    loss=loss/cfg.gradient_accumulation_steps
                                st3=0
                                loss.backward()
                                backward_time+=time.time()-st3
                                tr_loss += loss.item()
                                loss_uns+=loss.item()
                                loss1+=loss_ce.item()
                                if loss_kl!=0:
                                    loss2+=loss_kl.item()
                                torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                                torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                                epoch_step += 1
                                dial_batch=[]
                                epoch_step_uns+=1
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    logging.info("WARNING: ran out of memory during unsupervised train, batch idx:{}, batch size:{}, turn num:{}"\
                                        .format(batch_idx,len(dial_batch),len(dial_batch[0])))
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        with torch.cuda.device(self.device1):
                                            torch.cuda.empty_cache(self.device1)
                                        with torch.cuda.device(self.device2):
                                            torch.cuda.empty_cache(self.device2)
                                else:
                                    logging.info(str(exception))
                                    raise exception
                    
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    forward_time+=time.time()-st2
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    global_step+=1
                    loss_scalar = (tr_loss - logging_loss) / cfg.gradient_accumulation_steps
                    logging_loss = tr_loss
                    if self.tb_writer:
                        self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('loss', loss_scalar, global_step)
                else:
                    dial_batch_large=dial_batch_dict['batch']
                    dial_batch=[]
                    for i, dial in enumerate(dial_batch_large):
                        dial_batch.append(dial)
                        if len(dial_batch)==cfg.origin_batch_size or i==len(dial_batch_large)-1:
                            try:
                                self.PrioriModel.train()
                                self.PosteriorModel.train()
                                inputs_prior, labels_prior = self.reader.convert_batch_session(dial_batch,posterior_train=False)
                                inputs_posterior, labels_posterior = self.reader.convert_batch_session(dial_batch,posterior_train=True)
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                                outputs1 = self.PrioriModel(inputs_prior['contexts_tensor'])
                                loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                outputs2=self.PosteriorModel(inputs_posterior['contexts_tensor'])
                                loss_pos=self.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])

                                if cfg.loss_reg:
                                    loss_pri=loss_pri/cfg.gradient_accumulation_steps
                                    loss_pos=loss_pos/cfg.gradient_accumulation_steps
                                loss_pri.backward()
                                loss_pos.backward()
                                tr_loss+=loss_pri.item()+loss_pos.item()
                                loss_sup+=loss_pri.item()+loss_pos.item()
                                
                                torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                                torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                                dial_batch=[]
                                epoch_step+=1
                                epoch_step_sup += 1
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    logging.info("WARNING: ran out of memory during supervised train, batch idx:{}, batch size:{}, turn num:{}"\
                                        .format(batch_idx,len(dial_batch),len(dial_batch[0])))
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        with torch.cuda.device(self.device1):
                                            torch.cuda.empty_cache(self.device1)
                                        with torch.cuda.device(self.device2):
                                            torch.cuda.empty_cache(self.device2)
                                else:
                                    logging.info(str(exception))
                                    raise exception
                            
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    global_step+=1
                    loss_scalar = (tr_loss - logging_loss) / cfg.gradient_accumulation_steps
                    logging_loss = tr_loss
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss', loss_scalar, global_step)
 
            if epoch==0:
                logging.info('sup steps:{}, uns steps:{}'.format(epoch_step_sup,epoch_step_uns))

            logging.info('Epoch: {}, Train epoch time: {} min, generation time:{} min, forward time:{} min, backward time:{} min, loss:{}, loss_sup:{}, loss_uns:{}'.format(epoch, (time.time()-btm)/60, gen_time/60, forward_time/60, backward_time/60, tr_loss/epoch_step, loss_sup/epoch_step_sup,loss_uns/epoch_step_uns))
            if self.tb_writer:
                self.tb_writer.add_scalar('loss_sup',loss_sup/epoch_step_sup,epoch)
                self.tb_writer.add_scalar('loss_uns',loss_uns/epoch_step_uns,epoch)
                self.tb_writer.add_scalar('loss_ce',loss1/epoch_step_uns,epoch)
                self.tb_writer.add_scalar('loss_kl',loss2/epoch_step_uns,epoch)
            
            if cfg.evaluate_during_training:
                eval_loss=self.eval(model=self.PrioriModel)
                logging.info('Prior model evaluation loss:{}'.format(eval_loss))
                eval_result=self.validate_fast(data='dev')
                if self.tb_writer:
                    self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                    self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                    self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                    self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                    self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                
                if eval_result['score']>max_score:
                    max_score=eval_result['score']
                    self.save_model(path='best_score_model')
                    early_stop_count=cfg.early_stop_count
                else:
                    weight_decay_count-=1
                    if weight_decay_count==0 and not cfg.use_scheduler:
                        lr=lr*cfg.lr_decay
                        for group in optimizer1.param_groups:
                            group['lr'] = lr
                        for group in optimizer2.param_groups:
                            group['lr'] = lr
                        logging.info("learning rate decay to {}".format(lr))
                        weight_decay_count = cfg.weight_decay_count
                    if epoch>=warmup_epochs:
                        early_stop_count-=1
                        logging.info('early stop count:%d'%early_stop_count)
                if lr<1e-9 and not cfg.use_scheduler:
                    logging.info('learning rate too small, break')
                    break
                if early_stop_count==0 and cfg.early_stop:
                    logging.info('early stopped')
                    break
            else:
                if loss1/epoch_step<min_loss1:
                    min_loss1=loss1/epoch_step
                    self.save_model()
                if loss2/epoch_step<min_loss2:
                    min_loss2=loss2/epoch_step
                    self.save_model(posterior=True)
    
    def gen_turn_hidden_state(self, turn_batch, pv_batch, posterior=True, pv_bspn_batch=None, turn_domain_batch=None):
        if posterior:
            self.model=self.PosteriorModel
        else:
            self.model=self.PrioriModel
        self.model.eval()
        max_len_b=60
        max_len_a=20
        with torch.no_grad():
            # generate bspn
            contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn', posterior=posterior)
            bspn_batch=self.generate_batch(self.model, contexts, max_len_b, self.eos_b_id)
            if cfg.use_true_domain_for_ctr_train:
                bs_gen, db_gen, _, _=self.reader.get_bspn(bspn_batch, return_db=True, turn_domain=turn_batch['turn_domain'])
            else:
                bs_gen, _, _, _=self.reader.get_bspn(bspn_batch)
                turn_domain_batch, db_gen=self.get_turn_domain(turn_domain_batch, bs_gen, pv_bspn_batch)
            # generate aspn
            contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
                bspn_gen=bs_gen,db_gen=db_gen, posterior=posterior)
            aspn_batch=self.generate_batch(self.model, contexts, max_len_a, self.eos_a_id)
            aspn_gen=self.reader.get_aspn(aspn_batch)
            turn_batch['bspn']=bs_gen
            turn_batch['db']=db_gen
            turn_batch['aspn']=aspn_gen
            pv_batch=self.reader.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'], bs_gen)
        return turn_batch, pv_batch, bs_gen, turn_domain_batch
    
    def gen_session_hidden_state(self,original_batch,model_name=None,validate=False):
        if model_name=='PosteriorModel':
            self.model=self.PosteriorModel
        elif model_name=='PrioriModel':
            self.model=self.PrioriModel

        if cfg.mode=='test_pos' or validate or model_name=='PosteriorModel':
            prior=False
        else:
            prior=True
        self.model.eval()
        device=self.model.device
        max_len=60
        max_len_a=20
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        eos_a_id=self.eos_a_id
        with torch.no_grad():
            context_len=self.get_max_len(original_batch)
            if context_len>900:
                logging.info('The max length of current batch is:{}, we divide it by half'.format(context_len))
                batch_size=len(original_batch)
                batches=[ original_batch[:batch_size//2], original_batch[batch_size//2:] ]
            else:
                batches=[ original_batch ]
            new_batch=[]
            for batch in batches:
                try:
                    batch_size=len(batch)
                    #print('batch size:{}, turn_num:{}'.format(batch_size,len(batch[0])))
                    contexts=[[] for i in range(len(batch))]
                    resp=[[] for i in range(len(batch))]
                    aspn_gen=[[] for i in range(len(batch))]
                    bs_gen=[]
                    for turn_num in range(len(batch[0])):
                        past_key_values=None
                        end_flag=np.zeros(len(batch))
                        contexts=self.reader.convert_eval_batch_session(batch,contexts,turn_num,bs_gen,\
                            prior=prior,resp_gen=resp,aspn_gen=aspn_gen)
                        inputs,attentions=self.batch_align(contexts,left_len=max_len,return_attn=True)
                        inputs=torch.tensor(inputs).to(device)
                        attentions=torch.tensor(attentions).to(device)
                        if self.global_output>0 and cfg.example_log:
                            logging.info('generation examples:')
                            logging.info(self.tokenizer.decode(contexts[0]))
                            self.global_output-=1
                        for i in range(max_len):
                            position_ids = attentions.long().cumsum(-1) - 1
                            position_ids.masked_fill_(attentions == 0, 1)
                            if past_key_values is not None:
                                position_ids=position_ids[:, -1].unsqueeze(-1)
                            outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,
                                return_dict=True,use_cache=True,past_key_values=past_key_values)#B,T,V
                            past_key_values=outputs.past_key_values
                            if cfg.sample_type=='top1':
                                preds=outputs.logits[:,-1,:].argmax(-1)#B
                            elif cfg.sample_type=='topk':
                                prob=F.softmax(outputs.logits[:,-1,:],dim=-1)#B,V
                                topk_probs, topk_words = torch.topk(prob, cfg.topk_num)#B,topk_num
                                widx = torch.multinomial(topk_probs, 1, replacement=True)#B,1
                                preds = torch.gather(topk_words, 1, widx).squeeze()#B
                            if i==0:
                                bs_tensor=preds.unsqueeze(1)
                            else:
                                bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                            inputs=preds.unsqueeze(1)
                            attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                            end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                            if sum(end_flag==0)==0:
                                break
                        bs_gen,db_gen=self.reader.get_bspn(bs_tensor,return_db=True,data=batch,turn_num=turn_num)

                        contexts=self.reader.convert_eval_batch_session(batch,contexts,turn_num,bs_gen,prior=prior,db_gen=db_gen)
                        if cfg.model_act:
                            past_key_values=None
                            end_flag=np.zeros(len(batch))
                            #note that the left_len should be max_len_a, but i set it to max_len to reduce the case of out of memory
                            inputs,attentions=self.batch_align(contexts,left_len=max_len,return_attn=True)
                            inputs=torch.tensor(inputs).to(device)
                            attentions=torch.tensor(attentions).to(device)
                            if self.global_output>0 and cfg.example_log:
                                logging.info('generation examples:')
                                logging.info(self.tokenizer.decode(contexts[0]))
                            for i in range(max_len_a):
                                position_ids = attentions.long().cumsum(-1) - 1
                                position_ids.masked_fill_(attentions == 0, 1)
                                if past_key_values is not None:
                                    position_ids=position_ids[:, -1].unsqueeze(-1)
                                outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,
                                    return_dict=True,use_cache=True,past_key_values=past_key_values)#B,T,V
                                past_key_values=outputs.past_key_values
                                if cfg.sample_type=='top1':
                                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                                elif cfg.sample_type=='topk':
                                    prob=F.softmax(outputs.logits[:,-1,:],dim=-1)#B,V
                                    topk_probs, topk_words = torch.topk(prob, cfg.topk_num)#B,topk_num
                                    widx = torch.multinomial(topk_probs, 1, replacement=True)#B,1
                                    preds = torch.gather(topk_words, 1, widx).squeeze()#B
                                if i==0:
                                    bs_tensor=preds.unsqueeze(1)
                                else:
                                    bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                                inputs=preds.unsqueeze(1)
                                attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                                end_flag+=(preds.cpu().numpy()==eos_a_id).astype(float)
                                if sum(end_flag==0)==0:
                                    break
                            aspn_gen=self.reader.get_aspn(bs_tensor)

                        for i in range(len(batch)):
                            if validate:
                                batch[i][turn_num]['bspn_gen']=bs_gen[i]
                                batch[i][turn_num]['db_gen']=db_gen[i]
                                if cfg.model_act:
                                    batch[i][turn_num]['aspn_gen']=aspn_gen[i]
                            else:
                                batch[i][turn_num]['bspn']=bs_gen[i]
                                batch[i][turn_num]['db']=db_gen[i]
                                if cfg.model_act:
                                    batch[i][turn_num]['aspn']=aspn_gen[i]
                            if cfg.model_act:
                                resp[i]=batch[i][turn_num]['aspn']+batch[i][turn_num]['resp']#take aspn and resp as one resp
                            else:
                                resp[i]=batch[i][turn_num]['resp']
                    new_batch+=batch
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logging.info("WARNING: ran out of memory during generation, and the batch will be divided half, batch size:{}, turn num:{}"\
                            .format(len(batch),len(batch[0])))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(device):
                                torch.cuda.empty_cache()
                        #current batch out of memory, split it half
                        batches+= [ batch[:len(batch)//2], batch[len(batch)//2:] ]
                    else:
                        logging.info(str(exception))
                        raise exception

        return new_batch

    
    def get_turn_ST_input(self,inputs, logits, labels1, labels2):
        #add straight through for variational learning
        #inputs:B,T1
        #logits:B,T1,V or B,T2,V
        #labels1:B,T1
        #labels2:B,T1 or B,T2
        onehot=F.one_hot(inputs,self.vocab_size).float() # B, T, V
        resp_label=cfg.pad_id*torch.ones(labels1.shape).long().to(labels1.device)
        for dial_idx in range(logits.size(0)):
            label_pri=labels1[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist() #0 for pad token and 1 for hidden states tokens
            label_post=labels2[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            st_idx=label_post.index(1)
            h_len=len(label_post)-label_post[::-1].index(1)-st_idx
            probs=F.softmax(logits[dial_idx, st_idx:st_idx+h_len-1,:], dim=-1) # probs of hidden states
            st_idx=label_pri.index(1)
            onehot[dial_idx, st_idx+1:st_idx+h_len, :]+=(probs-probs.detach()).to(onehot.device)
            resp_label[dial_idx, st_idx+h_len:]=labels1[dial_idx, st_idx+h_len:]
        return onehot, resp_label
    
    def get_session_ST_input(self,inputs,logits,labels1,labels2):
        #inputs:B,T1
        #logits:B,T1,V or B,T2,V
        #labels1:B,T1
        #labels2:B,T1 or B,T2
        onehot=F.one_hot(inputs,self.vocab_size).float()
        for dial_idx in range(logits.size(0)):
            label_pri=labels1[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()#0 for pad token and 1 for hidden states tokens
            label_post=labels2[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            label_pri.reverse()#Traverse from back to front
            label_post.reverse()
            loc1=0
            loc2=0
            loc3=0
            loc4=0
            while(1):
                if 1 not in label_pri:
                    break
                loc1=label_pri.index(1)+loc2
                label_pri=label_pri[loc1-loc2:]
                if 0 not in label_pri:
                    break
                loc2=label_pri.index(0)+loc1
                if 1 not in label_post:
                    break
                loc3=label_post.index(1)+loc4
                label_post=label_post[loc3-loc4:]
                if 0 not in label_post:
                    break
                loc4=label_post.index(0)+loc3
                if (loc4-loc3)!=(loc2-loc1):
                    print('location:',loc1,loc2,loc3,loc4)
                assert loc4-loc3==loc2-loc1
                probs=F.softmax(logits[dial_idx,-loc4:-loc3-1,:])
                if loc1==0:
                    onehot[dial_idx,-loc2+1:,:]+=(probs-probs.detach()).to(onehot.device)
                else:
                    onehot[dial_idx,-loc2+1:-loc1,:]+=(probs-probs.detach()).to(onehot.device)
                label_pri=label_pri[loc2-loc1:]
                label_post=label_post[loc4-loc3:]
        return onehot


    def kl_loss(self, p_proba, q_proba): # [B, T, V] or [T,V]
        dim=p_proba.dim()
        loss = q_proba * (torch.log(q_proba+self.eps) - torch.log(p_proba+self.eps))
        loss = torch.sum(loss, dim=-1)   # sum over vocabulary
        loss = torch.sum(loss, dim=-1)   # sum over sequence
        if dim==2:
            return loss
        else:
            return loss.mean()

    def get_turn_kl_loss(self,logits_pri,logits_post,labels_pri,labels_post):
        # logits_pri:B,T1,V
        # logits_post:B,T2,V
        # labels_pri:B,T1. bspn's label in prior sequence
        # labels_post:B,T2. bspn's label in posterior sequence
        # what labels do is to find the logits corresponding to bspn
        loss=0
        count=0
        for dial_idx in range(logits_pri.size(0)):
            label_pri=labels_pri[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist() #pad_id处为0，bspn为1
            label_post=labels_post[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            h_len=len(label_post)-label_post[::-1].index(1)-label_post.index(1)
            idx1=label_pri.index(1)
            idx2=label_post.index(1)
            probs_pri=F.softmax(logits_pri[dial_idx, idx1:idx1+h_len-1,:],dim=-1)
            probs_post=F.softmax(logits_post[dial_idx, idx2:idx2+h_len-1,:],dim=-1)
            loss+=self.kl_loss(probs_pri,probs_post.to(probs_pri.device))
            count+=h_len
        return loss/count

    def get_session_kl_loss(self,logits_pri,logits_post,labels_pri,labels_post):
        # logits_pri:B,T1,V
        # logits_post:B,T2,V
        # labels_pri:B,T1. bspn's label in prior sequence
        # labels_post:B,T2. bspn's label in posterior sequence
        # what labels do is to find the logits corresponding to bspn
        loss=0
        count=0
        for dial_idx in range(logits_pri.size(0)):
            label_pri=labels_pri[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()#pad_id处为0，bspn为1
            label_post=labels_post[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            label_pri.reverse()#从后往前遍历
            label_post.reverse()
            turn_count=0
            loc1=0
            loc2=0
            loc3=0
            loc4=0
            while(1):
                if 1 not in label_pri:
                    break
                loc1=label_pri.index(1)+loc2
                label_pri=label_pri[loc1-loc2:]
                if 0 not in label_pri:
                    break
                loc2=label_pri.index(0)+loc1
                if 1 not in label_post:
                    break
                loc3=label_post.index(1)+loc4
                label_post=label_post[loc3-loc4:]
                if 0 not in label_post:
                    break
                loc4=label_post.index(0)+loc3
                bspn_len=min(loc2-loc1,loc4-loc3)
                probs_pri=F.softmax(logits_pri[dial_idx,-(loc1+bspn_len):-loc1-1,:],dim=-1)
                probs_post=F.softmax(logits_post[dial_idx,-(loc3+bspn_len):-loc3-1,:],dim=-1)
                loss+=self.kl_loss(probs_pri,probs_post.to(probs_pri.device))
                count+=bspn_len
                turn_count+=1
                label_pri=label_pri[loc2-loc1:]
                label_post=label_post[loc4-loc3:]
        
        return loss/count


    def get_max_len(self,batch):
        max_len=0
        for dial in batch:
            dial_len=0
            for turn in dial:
                dial_len+=len(turn['user'])+len(turn['resp'])
            if dial_len>max_len:
                max_len=dial_len
        return max_len
    
    def get_turn_domain(self, turn_domain_batch, bs_batch, pv_bs_batch=None):

        db_batch=[]
        for i, bspn in enumerate(bs_batch):
            bspn_tokens=self.tokenizer.decode(bspn)
            cons=self.reader.bspan_to_constraint_dict(bspn_tokens)
            cur_domain=list(cons.keys())
            if len(cur_domain)==0:
                db_result = self.tokenizer.encode('<sos_db> [db_nores] <eos_db>')
            else:
                if len(cur_domain)==1:
                    turn_domain_batch[i]=cur_domain
                else:
                    if pv_bs_batch is None:
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                turn_domain_batch[i]=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(self.reader.bspan_to_constraint_dict(self.tokenizer.decode(pv_bs_batch[i])).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                turn_domain_batch[i]=[domain]
                db_result = self.reader.bspan_to_DBpointer(bspn_tokens, turn_domain_batch[i]) #[db_x]
                db_result = self.tokenizer.encode('<sos_db> '+ db_result + ' <eos_db>')
            db_batch.append(db_result)
        return turn_domain_batch, db_batch

    
    def generate_batch(self, model, contexts, max_len, eos_id, beam=1):
        # generate by batch
        # contexts: a list of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids with pre pad 
        batch_size=len(contexts)
        end_flag=np.zeros(batch_size)
        if beam>1:
            beam_box=[beam]*batch_size
            beam_result=[[] for _ in range(batch_size)]
            max_prob=[-float('inf')]*batch_size
        past_key_values=None
        inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                if beam==1:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    if inputs.size(0)==0:
                        raise ValueError(contexts, inputs.cpu().list(), attentions)
                    outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        gen_tensor=preds.unsqueeze(1)
                    else:
                        gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                else:
                    if i==0:
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)
                        past_key_values=[outputs.past_key_values]*beam
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx#B, beam
                    else:
                        for j in range(beam):
                            inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                            outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                            past_key_values[j]=outputs.past_key_values
                            log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                            beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                            if j==0:
                                prob_pool= beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                                id_pool=beam_idx
                            else:
                                prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)],-1) # B, beam*beam
                                id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                        beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                        beam_idx=torch.gather(id_pool, -1, temp_id)
                        temp_id=temp_id//beam
                        new_past_key_values=copy.deepcopy(past_key_values)
                        for b in range(batch_size):
                            gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                            for t in range(beam):
                                for l in range(6):
                                    new_past_key_values[t][l][:, b, :,:,:]=past_key_values[temp_id[b, t]][l][:, b, :, :, :]
                        past_key_values=new_past_key_values
                        #past_key_values=[past_key_values[t] for t in temp_id.cpu().list()]
                        gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx
                    for m in range(batch_size):
                        for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                            if eos_id in gen:
                                beam_box[m]-=1
                                avg_prob=pv_beam_prob[m][n]/len(gen)
                                beam_result[m].append((gen, avg_prob))
                                pv_beam_prob[m][n]=-float('inf')
                    # we do not break during beam search
                    #if not any(beam_box):
                     #   break
            
        if beam==1:
            return gen_tensor.cpu().tolist()
        else:
            for i, tup in enumerate(beam_result):
                beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
                beam_result[i]=[item[0] for item in beam_list[:beam]]
            return beam_result        


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
    parse_arg_cfg(args)
    if 'test' in args.mode:
        cfg.eval_load_path=cfg.gpt_path
    else:  # train
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './experiments_21' if cfg.dataset==1 else './experiments'
            if cfg.exp_no=='':
                if cfg.mode=='semi_VL':
                    cfg.exp_no='VL_'
                    cfg.exp_no += str(cfg.spv_proportion)

            cfg.exp_path = os.path.join(experiments_path, cfg.exp_no)
            if cfg.save_log and not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.eval_load_path = cfg.exp_path

    device=cfg.cuda_device
    cfg.divided_path=os.path.join(cfg.data_path,'divided_data{}.json'.format(cfg.spv_proportion))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Semi_Model(device)

    if args.mode =='semi_VL':
        if cfg.turn_level:
            m.semi_VL_turn_level()
        else:
            m.semi_VL_session_level()
    elif args.mode =='semi_VL':  # test
        logging.info('Load model from :{}'.format(cfg.eval_load_path))
        m.validate_fast('test')


if __name__ == "__main__":
    main()
