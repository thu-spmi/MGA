# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
from re import match
from turtle import width
from typing import Sequence
from config import global_config as cfg
from eval import MultiWozEvaluator, get_richness
from reader import MultiWozReader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json, random
import ontology
import torch
import numpy as np
from mwzeval.metrics import Evaluator
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import copy, re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from significance_test import matched_pair, McNemar
from session import turn_level_session
from collections import Counter
stopwords = ['and','are','as','at','be','been','but','by', 'for','however','if', 
             'not','of','on','or','so','the','there','was','were','whatever','whether','would']

tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS-100/best_score_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
std_evaluator=Evaluator(bleu=1, success=1, richness=0)

def compare_offline_result(path1, path2, show_num=10):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    dials1=evaluator.pack_dial(data1)
    dials2=evaluator.pack_dial(data2)
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    dial_id_list=random.sample(reader.test_list, show_num)
    dial_samples=[]
    for dial_id in dials1:
        dial1=dials1[dial_id]
        dial2=dials2[dial_id]
        if dial_id+'.json' in dial_id_list:
            dial_samples.append({'dial1':dial1, 'dial2':dial2})
        reqs = {}
        goal = {}
        if '.json' not in dial_id and '.json' in list(evaluator.all_data.keys())[0]:
            dial_id = dial_id + '.json'
        for domain in ontology.all_domains:
            if evaluator.all_data[dial_id]['goal'].get(domain):
                true_goal = evaluator.all_data[dial_id]['goal']
                goal = evaluator._parseGoal(goal, true_goal, domain)
        # print(goal)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']

        # print('\n',dial_id)
        success1, match1, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2))#, succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1))#, succ2_unsuc1)
    examples=[]
    for item in dial_samples:
        dialog=[]
        for turn1, turn2 in zip(item['dial1'], item['dial2']):
            if turn1['user']=='':
                continue
            entry={'user': turn1['user'], 'Oracle':turn1['resp'], 'Sup':turn1['resp_gen'], 'RL':turn2['resp_gen']}
            dialog.append(entry)
        examples.append(dialog)
    json.dump(examples, open('analysis/examples.json', 'w'), indent=2)

def find_special_case(path1, path2):
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    for dial_id, dial1 in data1.items():
        dial2=data2[dial_id]
        for turn1, turn2 in zip(dial1, dial2):
            pass

def compare_online_result(path1, path2):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    flag1=0
    flag2=0
    for i, dial_id in enumerate(reader.test_list):
        reqs = {}
        goal = {}
        dial1=data1[i]
        dial2=data2[i]
        if isinstance(dial1, list):
            data1[i]={dial_id:dial1}
            flag1=1
        elif isinstance(dial1, dict):
            dial1=dial1[dial_id]
        
        if isinstance(dial2, list):
            data2[i]={dial_id:dial2}
            flag2=1
        elif isinstance(dial2, dict):
            dial2=dial2[dial_id]

        init_goal=reader.data[dial_id]['goal']
        for domain in ontology.all_domains:
            if init_goal.get(domain):
                true_goal = init_goal
                goal = evaluator._parseGoal(goal, true_goal, domain)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']
        success1, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2), succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1), succ2_unsuc1)
    if flag1:
        json.dump(data1, open(path1, 'w'), indent=2)
    if flag2:
        json.dump(data2, open(path2, 'w'), indent=2)

def group_act(act):
    for domain in act:
        for intent, sv in act[domain].items():
            act[domain][intent]=set(sv)
    return act

def group_state(state):
    for domain, sv in state.items():
        state[domain]=set(sv)
    return state

def find_unseen_usr_act(path1=None, path2=None):
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    train_act_pool=[]
    unseen_act_pool=[]
    unseen_dials=[]
    for dial_id, dial in data.items():
        if dial_id in reader.train_list:
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    train_act_pool.append(user_act)
    for dial_id, dial in data.items():
        if dial_id in reader.test_list:# or dial_id in reader.dev_list:
            unseen_turns=0
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool.append(user_act)
                    unseen_turns+=1
            if unseen_turns>0:
                unseen_dials.append(dial_id)
    print('Total training acts:', len(train_act_pool), 'Unseen acts:', len(unseen_act_pool))
    print('Unseen dials:',len(unseen_dials))
    if path1 and path2:
        data1=json.load(open(path1, 'r', encoding='utf-8'))
        data2=json.load(open(path2, 'r', encoding='utf-8'))
        unseen_act_pool1=[]
        unseen_act_pool2=[]
        for dial1, dial2 in zip(data1, data2):
            dial1=list(dial1.values())[0]
            dial2=list(dial2.values())[0]
            for turn in dial1:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool1.append(user_act)
            for turn in dial2:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool2.append(user_act)
        print('Unseen acts in path1:', len(unseen_act_pool1))
        print('Unseen acts in path2:', len(unseen_act_pool2))
    return unseen_dials

def count_act(data):
    act_pool=[]
    for turn in data:
        if turn['user']=='':
            continue
        else:
            sys_act=reader.aspan_to_act_dict(turn['aspn_gen'], 'sys')
            sys_act=group_act(sys_act)
            if sys_act not in act_pool:
                act_pool.append(sys_act)
    print('Act num:', len(act_pool))

def count_online(data):
    user_act_pool=[]
    sys_act_pool=[]
    for dial in data:
        for turn in dial:
            user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
            user_act=group_act(user_act)
            if user_act not in user_act_pool:
                user_act_pool.append(user_act)
            sys_act=reader.aspan_to_act_dict(turn['aspn'], 'sys')
            sys_act=group_act(sys_act)
            if sys_act not in sys_act_pool:
                sys_act_pool.append(sys_act)
    print('Total sys act:', len(sys_act_pool), 'total user acts:', len(user_act_pool))

def count_state(data):
    state_pool=[]
    act_pool=[]
    for turn in data:
        if turn['user']=='':
            continue
        state=reader.bspan_to_constraint_dict(turn['bspn_gen'])
        state=group_state(state)
        act=reader.aspan_to_act_dict(turn['aspn_gen'], 'sys')
        act=group_act(act)
        if state not in state_pool:
            state_pool.append(state)
            act_pool.append([act])
        elif act not in act_pool[state_pool.index(state)]:
            act_pool[state_pool.index(state)].append(act)
    print('Total states:',len(state_pool), 'Average actions per state:', np.mean([len(item) for item in act_pool]))

def find_unseen_sys_act():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r', encoding='utf-8'))
    train_act_pool=[]
    unseen_act_pool=[]
    test_act_pool=[]
    unseen_dials={}
    for dial_id, dial in data.items():
        if dial_id in reader.train_list:
            for turn in dial:
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], 'sys')
                sys_act=group_act(sys_act)
                if sys_act not in train_act_pool:
                    train_act_pool.append(sys_act)
    for dial_id, dial in data.items():
        if dial_id in reader.test_list:# or dial_id in reader.dev_list:
            unseen_turns=[]
            for turn_id, turn in enumerate(dial):
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], 'sys')
                sys_act=group_act(sys_act)
                if sys_act not in test_act_pool:
                    test_act_pool.append(sys_act)
                if sys_act not in train_act_pool:
                    unseen_act_pool.append(sys_act)
                    unseen_turns.append(turn_id)
            if len(unseen_turns)>0:
                unseen_dials[dial_id]=unseen_turns
    print('Total training acts:', len(train_act_pool), 'test acts:',len(test_act_pool),'Unseen acts:', len(unseen_act_pool))
    print('Unseen dials:',len(unseen_dials))
    json.dump(unseen_dials, open('analysis/unseen_turns.json', 'w'), indent=2)

    return unseen_dials

def calculate_unseen_acc(unseen_turns, path1=None, path2=None):
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    total_unseen_act=0
    sup_acc=0
    rl_acc=0
    tp1=0
    fp1=0
    tp2=0
    fp2=0
    count=0
    for dial_id in unseen_turns:
        for t in unseen_turns[dial_id]:
            count+=1
    print('Total unseen act:', count)
    for turn1, turn2 in zip(data1, data2):
        dial_id=turn1['dial_id']+'.json'
        if dial_id in unseen_turns and turn1['user']!='' and turn1['turn_num'] in unseen_turns[dial_id]:
            total_unseen_act+=1
            #unseen_turns[dial_id]=unseen_turns[dial_id][1:]
            oracle_act=group_act(reader.aspan_to_act_dict(turn1['aspn'], side='sys'))
            sup_act=group_act(reader.aspan_to_act_dict(turn1['aspn_gen'], side='sys'))
            rl_act=group_act(reader.aspan_to_act_dict(turn2['aspn_gen'], side='sys'))
            if sup_act==oracle_act:
                sup_acc+=1
            if rl_act==oracle_act:
                rl_acc+=1
            for domain in sup_act:
                for intent, slots in sup_act[domain].items():
                    if domain not in oracle_act or intent not in oracle_act[domain]:
                        fp1+=len(slots)
                        continue
                    for slot in slots:
                        if slot in oracle_act[domain][intent]:
                            tp1+=1
                        else:
                            fp1+=1
            for domain in rl_act:
                for intent, slots in rl_act[domain].items():
                    if domain not in oracle_act or intent not in oracle_act[domain]:
                        fp2+=len(slots)
                        continue
                    for slot in slots:
                        if slot in oracle_act[domain][intent]:
                            tp2+=1
                        else:
                            fp2+=1
    print('Total unseen acts:{}, Sup acc:{}, RL acc:{}'.format(total_unseen_act, sup_acc, rl_acc))
    print(tp1, fp1, tp1/(tp1+fp1))
    print(tp2, fp2, tp2/(tp2+fp2))

def extract_goal():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    goal_list={}
    for dial_id, dial in data.items():
        goal=dial['goal']
        goal_list[dial_id]=goal
    json.dump(goal_list, open('analysis/goals.json', 'w'), indent=2)

def get_attentions(model_path, mode='bspn', encode_key=['user', 'bspn', 'db', 'aspn', 'resp'], turn_th=4):
    tok=GPT2Tokenizer.from_pretrained(model_path)
    model=GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    key_pool=['user', 'bspn', 'db', 'aspn', 'resp']
    num=len(encode_key)*(turn_th-1)+key_pool.index(mode)
    attention_list=[]
    count=0
    for dial_id, dial in data.items():
        dial_id=dial_id[:-5] if '.json' in dial_id else dial_id
        if dial_id not in reader.test_list:
            continue
        if len(dial['log'])<turn_th:
            continue
        for turn in dial['log']:
            for key, sent in turn.items():
                if key in key_pool:
                    turn[key]=tok.encode(sent)
        sequence=[]
        st_idx_list=[]
        ed_idx_list=[]
        flag=0
        for id, turn in enumerate(dial['log']):
            if id<turn_th-1:
                for key in encode_key:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            elif id==turn_th-1:
                for key in key_pool:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    if key==mode:
                        st_idx=id1
                        ed_idx=id2
                        flag=1
                        break
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            if flag:
                break
        assert len(st_idx_list)==num
        if len(sequence)>1024:
            continue
        with torch.no_grad():
            outputs=model.forward(torch.tensor([sequence]), return_dict=True, output_attentions=True)
        attentions=outputs.attentions[-1] #last layer
        #ed_idx=min(st_idx+max_len, ed_idx)
        attention=torch.mean(attentions[0, :, st_idx:ed_idx,], dim=0) # T_b, T
        avg1=torch.mean(attention, dim=0) #T
        entry=[]
        for id1, id2 in zip(st_idx_list, ed_idx_list):
            avg2=avg1[id1:id2].mean().item()
            if np.isnan(avg2):
                avg2=0
            entry.append(avg2)
        attention_list.append(entry)
        count+=1

        '''
        attention=attention[:,1:st_idx]
        attention/=attention.max()
        plt.figure()
        plt.imshow(attention.numpy(), cmap=plt.cm.hot)
        plt.xlabel('Previous information')
        plt.ylabel('Belief state')
        plt.colorbar()
        #plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        #plt.yticks(np.arange(ed_idx-st_idx-1))
        plt.title('Attentions')
        plt.savefig('analysis/attention.png')
        #plt.show()
        break
        '''
    print('Count dials:', count)
    print(len(attention_list))
    avg_attentions=list(np.mean(attention_list, axis=0))
    print(avg_attentions)
    print(list(np.var(attention_list, axis=0)))
    return avg_attentions

def get_attentions1(model_path, mode='bspn', encode_key=['bspn','resp'], turn_th=4):
    tok=GPT2Tokenizer.from_pretrained(model_path)
    model=GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    key_pool=['user', 'bspn', 'db', 'aspn', 'resp']
    num=len(encode_key)+key_pool.index(mode)
    attention_list=[]
    count=0
    for dial_id, dial in data.items():
        dial_id=dial_id[:-5] if '.json' in dial_id else dial_id
        if dial_id not in reader.test_list:
            continue
        if len(dial['log'])<turn_th:
            continue
        for turn in dial['log']:
            for key, sent in turn.items():
                if key in key_pool:
                    turn[key]=tok.encode(sent)
        sequence=[]
        st_idx_list=[]
        ed_idx_list=[]
        flag=0
        for id, turn in enumerate(dial['log']):
            if id==turn_th-2:
                for key in encode_key:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            elif id==turn_th-1:
                for key in key_pool:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    if key==mode:
                        st_idx=id1
                        ed_idx=id2
                        flag=1
                        break
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            if flag:
                break
        assert len(st_idx_list)==num
        if len(sequence)>1024:
            continue
        with torch.no_grad():
            outputs=model.forward(torch.tensor([sequence]), return_dict=True, output_attentions=True)
        attentions=outputs.attentions[-1] #last layer
        #ed_idx=min(st_idx+max_len, ed_idx)
        attention=torch.mean(attentions[0, :, st_idx:ed_idx,], dim=0) # T_b, T
        avg1=torch.mean(attention, dim=0) #T
        entry=[]
        for id1, id2 in zip(st_idx_list, ed_idx_list):
            avg2=avg1[id1:id2].mean().item()
            if np.isnan(avg2):
                avg2=0
            entry.append(avg2)
        attention_list.append(entry)
        count+=1
    print('Count dials:', count)
    print(len(attention_list))
    print(list(np.mean(attention_list, axis=0)))
    print(list(np.var(attention_list, axis=0)))

def find_attention_case(model_path, mode='bspn', encode_key=['user', 'bspn', 'db', 'aspn', 'resp'], turn_th=4):
    print('******** Attention heatmap ************')
    tok=GPT2Tokenizer.from_pretrained(model_path)
    model=GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    key_pool=['user', 'bspn', 'db', 'aspn', 'resp']
    count=0
    for dial_id, dial in data.items():
        dial_id=dial_id.strip('.json')
        if dial_id not in reader.test_list:
            continue
        if len(dial['log'])<turn_th:
            continue
        if count!=5:
            count+=1
            continue
        for turn in dial['log']:
            for key, sent in turn.items():
                if key in key_pool:
                    turn[key]=reader.modified_encode(sent, tok)
        sequence=[]
        st_idx_list=[]
        ed_idx_list=[]
        flag=0
        for id, turn in enumerate(dial['log']):
            if id<turn_th-1:
                for key in encode_key:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]                  
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            elif id==turn_th-1:
                for key in key_pool:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    if key==mode:
                        st_idx=id1
                        ed_idx=id2
                        flag=1
                        break
                    #st_idx_list.append(id1)
                    #ed_idx_list.append(id2)
            if flag:
                break
        print('Previous variables:', len(st_idx_list))
        if len(sequence)>1024:
            continue
        with torch.no_grad():
            outputs=model.forward(torch.tensor([sequence]), return_dict=True, output_attentions=True)
        attentions=outputs.attentions[-1] #last layer
        #ed_idx=min(st_idx+max_len, ed_idx)
        attentions=torch.mean(attentions[0, :, st_idx:ed_idx,], dim=0) # T_b, T
        bs=tok.convert_ids_to_tokens(sequence[st_idx:ed_idx])
        bs=[item.strip('Ġ') for item in bs]
        for i, (id1, id2) in enumerate(zip(st_idx_list, ed_idx_list)):
            #if id1==id2:
             #   continue
            key=encode_key[i%len(encode_key)]
            key_id=key[0] if key!='db' else 'db'
            sos_id=['<sos_{}>'.format(key_id)]
            eos_id=['<eos_{}>'.format(key_id)]
            temp=torch.zeros(attentions.size(0), 1)
            if i==0:
                attention=torch.cat([temp, attentions[:,id1+1:id2+1], temp],dim=1)
                pv_bs=sos_id+tok.convert_ids_to_tokens(sequence[id1:id2])+eos_id
            else:
                attention=torch.cat([attention, temp, attentions[:,id1+1:id2+1], temp], dim=1)
                pv_bs+=sos_id+tok.convert_ids_to_tokens(sequence[id1:id2])+eos_id
        pv_bs=[item.strip('Ġ').replace('[value_', '[') for item in pv_bs]
        '''
        plt.figure(figsize=(len(pv_bs1), len(bs)))
        ax=plt.gca()
        im=ax.imshow(attention1.numpy(), cmap=plt.cm.hot)
        # recttuple (left, bottom, right, top) default: (0, 0, 1, 1)
        # 0: most left or most bottom
        # 1: most right or most top
        plt.tight_layout(rect=(0.1, 0.1, 0.9, 1))
        plt.xticks(np.arange(len(pv_bs1)), pv_bs1, rotation=90, fontsize=7)
        plt.yticks(np.arange(len(bs)),labels=bs, fontsize=8)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #plt.title('Attentions')
        plt.savefig('analysis/heatmap_u.png')
        print('***Figure saved***')
        '''
        pointer=0
        step=65
        while(pointer<len(pv_bs)):
            plt.figure(figsize=(20, len(bs)))
            ax=plt.gca()
            im=ax.imshow(attention[:, pointer:pointer+step].numpy(), cmap=plt.cm.hot)
            plt.tight_layout(rect=(0.1, 0.1, 0.9, 1))
            plt.xticks(np.arange(len(pv_bs[pointer:pointer+step])), pv_bs[pointer:pointer+step], rotation=90, fontsize=16)
            plt.yticks(np.arange(len(bs)),labels=bs, fontsize=15)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            #plt.title('Attentions')
            plt.savefig('analysis/heatmap_b_{}_{}.png'.format(count, pointer))
            pointer+=step
        count+=1
        print('***Figure saved***')
        print(' '.join(pv_bs))
        '''
        for i, (id1, id2) in enumerate(zip(st_idx_list, ed_idx_list)):
            if id1==id2:
                continue
            attention=attentions[:,id1+1:id2+1]
            pv_bs=tok.convert_ids_to_tokens(sequence[id1:id2])
            pv_bs=[item.strip('Ġ') for item in pv_bs]
            print(pv_bs)
            #attention/=attention.max()
            plt.figure()
            plt.imshow(attention.numpy(), cmap=plt.cm.hot)
            #plt.xlabel('${b_%d}$'%(i+1))
            #plt.ylabel('${b_%d}$'%(i+2))
            plt.colorbar()
            plt.tight_layout(rect=(0.25, 0.25, 1, 1))
            plt.xticks(np.arange(len(pv_bs)), pv_bs, rotation=90)
            plt.yticks(np.arange(len(bs)),labels=bs)
            #plt.title('Attentions')
            plt.savefig('analysis/attention_%d.png'%(i+1))
        '''
        if count==10:
            break

def length_statistics():
    data=json.load(open('data/multi-woz-2.1-processed/new_db_se_blank_encoded.data.json', 'r', encoding='utf-8'))
    #session-level
    total=0
    exceed=0
    mean_len, max_len=0, 0
    len_list=[]
    for dial in data['train']:
        length=0
        total+=1
        for turn in dial:
            length+=len(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
        if length>1024:
            exceed+=1
            print(length)
        len_list.append(length)
        mean_len+=length/len(data['train'])
        if length>max_len:
            max_len=length
    print('Total training sequences:', total, 'sequences exceeding limit:', exceed)
    print('Mean length:{}, max length:{}'.format(mean_len, max_len))
    print(np.mean(len_list), np.sqrt(np.var(len_list)))
    #turn-level
    total=0
    exceed=0
    mean_len, max_len=0, 0
    len_list=[]
    total_turn=sum([len(dial) for dial in data['train']])
    for dial in data['train']:
        total+=1
        history_len=0
        for turn in dial:
            total+=1
            length = history_len+len(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
            #history_len+=len(turn['user']+turn['resp'])
            history_len=len(turn['bspn']+turn['resp'])
            if length>1024:
                exceed+=1
            len_list.append(length)
            mean_len+=length/total_turn
            if length>max_len:
                max_len=length
    print('Total training sequences:', total, 'sequences exceeding limit:', exceed)
    print('Mean length:{}, max length:{}'.format(mean_len, max_len))
    print(np.mean(len_list), np.sqrt(np.var(len_list)))

def prepare_for_std_eval(path=None, data=None):
    if path:
        data=json.load(open(path, 'r', encoding='utf-8'))
    new_data={}
    dials=evaluator.pack_dial(data)
    for dial_id in dials:
        new_data[dial_id]=[]
        dial=dials[dial_id]
        for turn in dial:
            if turn['user']=='':
                continue
            entry={}
            entry['response']=turn['resp_gen']
            entry['state']=reader.bspan_to_constraint_dict(turn['bspn_gen'])
            new_data[dial_id].append(entry)
    if path:
        new_path=path[:-5]+'std.json'
        json.dump(new_data, open(new_path, 'w'), indent=2)
    return new_data

def get_metrics_list(path, prepared=False, dial_order=None):
    results=json.load(open(path, 'r'))
    input_data=prepare_for_std_eval(data=results) if not prepared else results
    if dial_order:
        new_data={}
        for dial_id in dial_order:
            if dial_id not in input_data:
                print('No dial id:', dial_id)
                continue
            new_data[dial_id]=input_data[dial_id]
        input_data=new_data
    results, match_list, success_list, bleu_list = std_evaluator.evaluate(input_data, return_all=True)
    print(results)
    return match_list, success_list, bleu_list, list(input_data.keys())

def compare_list(list1, list2):
    c1=0
    c2=0
    for t1, t2 in zip(list1, list2):
        if t1 and not t2:
            c1+=1
        elif not t1 and t2:
            c2+=1
    print(c1,c2)

def get_nooffer_slot():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r'))
    nooffer_dict={}
    for dial in data.values():
        for turn in dial:
            if '[nooffer]' in turn['sys_act']:
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], side='sys')
                for domain in sys_act:
                    if 'nooffer' in sys_act[domain]:
                        if domain not in nooffer_dict:
                            nooffer_dict[domain]=[]
                        for slot in sys_act[domain]['nooffer']:
                            if slot not in nooffer_dict[domain]:
                                nooffer_dict[domain].append(slot)
    print(nooffer_dict)

def compare_init_goal():
    data0=json.load(open('data/multi-woz-2.1-processed/data_for_us0.json', 'r'))
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r'))
    count, total = 0, 0
    total_slots, unequal_slots=0, 0
    for dial_id, dial in data.items():
        init_goal=reader.aspan_to_act_dict(dial[0]['goal'], side='user')
        init_goal0=reader.aspan_to_act_dict(data0[dial_id][0]['goal'], side='user')
        not_equal=False
        total+=1
        for domain in init_goal:
            if domain not in init_goal0:
                not_equal=True
                continue
            for intent in init_goal[domain]:
                if intent not in init_goal0[domain]:
                    not_equal=True
                    continue
                if isinstance(init_goal[domain][intent],dict):
                    if init_goal[domain][intent]!=init_goal0[domain][intent]:
                        not_equal=True
                        continue
                elif isinstance(init_goal[domain][intent], list):
                    if set(init_goal[domain][intent])!=set(init_goal0[domain][intent]):
                        not_equal=True
                        continue
        if not not_equal:
            count+=1
    print('Equal initial goal:', count, 'total:', total)

def analyze_unsuc_dial():
    data=json.load(open('analysis/gen_dials_100_unsuc.json', 'r'))      
    count=0
    for dial in data:
        flag=0
        for turn in dial['log']:
            #if 'not looking to make a booking' in turn['user']:
            if 'guest house'  in turn['user']:
                flag=1
                print()
        if flag:
            count+=1
    print(count)

def evaluate_dialog_with_ABUS(path):
    data=json.load(open(path, 'r'))
    success, match = 0, 0
    turn_num=0
    for dial_id, dial in data.items():
        s, m = evaluator.get_metrics(dial['goal'], dial['log'])
        success+=s
        match+=m
        turn_num+=len(dial['log'])
    print(success/len(data), match/len(data), turn_num/len(data))

def extract_goal_from_ABUS(path):
    data=json.load(open(path, 'r'))
    goal_list=[]
    for dial_id, dial in data.items():
        goal=dial['goal']
        new_goal=reader.unify_goal(goal)
        goal_list.append(new_goal)
    json.dump(goal_list, open('analysis/goal_list.json', 'w'), indent=2)

def plot_ubar_attention_bar(attentions, turn_num):
    ticks=[]
    for i in range(1, turn_num):
        for key in ['u', 'b', 'db', 'a', 'r']:
            tick='${}_{}$'.format(key, i)
            ticks.append(tick)
    ticks+=['$u_{}$'.format(turn_num)]
    #plt.ylabel('Average Attention Weight', fontsize='large')
    plt.figure(figsize=(2*turn_num,6))
    plt.title('Average Attention Weights of $b_{}$'.format(turn_num), fontsize='xx-large')
    plt.bar(np.arange(len(attentions)), attentions)
    plt.xticks(np.arange(len(attentions)), ticks, fontsize=13)
    plt.yticks(fontsize=14)
    plt.savefig('analysis/attention_ubar_turn{}.png'.format(turn_num))

def plot_mga_attention_bar(attentions, turn_num):
    attn1=[0.01677419327128426, 0.0035091243116767146, 0.00940846927366338]
    attn2=[0.0025351992187359553, 0.003397539328049742, 0.0008036897877869798, 0.001030863723811354, 0.0006276396838753283, 0.0013040142599585755, 0.0026612921946083186, 0.0007526683455087211, 0.0009952793496853763, 0.0007723224356811705, 0.0020394764483402683, 0.003982606828947733, 0.0015579156053286555, 0.002191299707623854, 0.0017771614970793746, 0.007519574391693114]
    ticks1=['$b_3$', '$r_3$', '$u_4$']
    ticks2=[]
    for i in range(1, 4):
        for key in ['u', 'b', 'db', 'a', 'r']:
            tick='${}_{}$'.format(key, i)
            ticks2.append(tick)
    ticks2+=['$u_4$']
    plt.figure()
    plt.bar(np.arange(len(attn1)), attn1, width=0.3)
    plt.xticks(np.arange(len(attn1)), ticks1, fontsize=13)
    plt.yticks(fontsize=14)
    plt.title('Average Attention Weights of $b_4$', fontsize='xx-large')
    plt.savefig('analysis/attention_b.png')
    #plt.ylabel('Average Attention Weight', fontsize='large')
    plt.figure()
    plt.title('Average Attention Weights of $b_4$', fontsize='xx-large')
    plt.bar(np.arange(len(attn2)), attn2)
    plt.xticks(np.arange(len(attn2)), ticks2, fontsize=13)
    plt.yticks(fontsize=14)
    plt.savefig('analysis/attention_b_ubar.png')

def merge_dial(dial1, dial2):
    dial_len=max(len(dial1), len(dial2))
    new_dial=[]
    keys=['pv_aspn', 'gpan', 'usr_act', 'user', 'bspn', 'db', 'aspn', 'resp']
    for i in range(dial_len):
        entry={}
        if i <len(dial1) and i<len(dial2):
            for key in keys:
                if i==0 and key=='pv_aspn':
                    continue
                entry[key+'1']=dial1[i][key]
                entry[key+'2']=dial2[i][key]
        elif i>=len(dial1) and i<len(dial2):
            for key in keys:
                entry[key+'2']=dial2[i][key]
        elif i<len(dial1) and i>=len(dial2):
            for key in keys:

                entry[key+'1']=dial1[i][key]
        new_dial.append(entry)
    return new_dial
            
def collect_improvement():
    res2=json.load(open('/home/liuhong/myworkspace/RL_exp/RL-3-30-ABUS/best_DS/validate_result.json', 'r'))
    res1=json.load(open('/mnt/workspace/liuhong/RL_exp/RL-4-24-gen-goal-only-ds/best_DS/validate_result.json', 'r'))
    count=0
    count1, count2=0, 0
    Improve_dials=[]
    Degrade_dials=[]
    for dial1, dial2 in zip(res1, res2):
        idx1=dial1[0]['DS_reward'].index('success_reward')+17
        suc1=dial1[0]['DS_reward'][idx1]
        idx2=dial2[0]['DS_reward'].index('success_reward')+17
        suc2=dial2[0]['DS_reward'][idx2]
        count+=1
        if suc1=='1' and suc2!='1':
            count1+=1
            Improve_dials.append(merge_dial(dial1, dial2))
        elif suc1!='1' and suc2=='1':
            count2+=1
            Degrade_dials.append(merge_dial(dial1, dial2))
    print('Total dials: ', count)
    print('Total improvement: ', count1)
    print('Total degradation: ', count2)
    json.dump(Improve_dials, open('analysis/improve_dials.json', 'w'), indent=2)
    json.dump(Degrade_dials, open('analysis/degrade_dials.json', 'w'), indent=2)

def temp():
    res1=json.load(open('/mnt/workspace/liuhong/RL_exp/RL-4-13-gen-goal/best_DS/validate_result.json', 'r'))
    res2=json.load(open('/mnt/workspace/liuhong/RL_exp/RL-4-13-gen-goal/best_DS/validate_result1.json', 'r'))
    count=0
    count1, count2=0, 0
    Improve_dials=[]
    Degrade_dials=[]
    for dial1, dial2 in zip(res1, res2):
        idx1=dial1[0]['DS_reward'].index('success_reward')+17
        suc1=dial1[0]['DS_reward'][idx1]
        idx2=dial2[0]['DS_reward'].index('success_reward')+17
        suc2=dial2[0]['DS_reward'][idx2]
        count+=1
        if suc1=='1':
            count1+=1
        if suc2=='1':
            count2+=1
    print('Total dials: ', count)
    print('Total improvement: ', count1)
    print('Total degradation: ', count2)

def get_diversity(path):
    result=json.load(open(path, 'r'))
    print(len(result))
    gen=[]
    for dial in result:
        for turn in dial:
            gen.append(turn['user'])
    print(get_richness(gen))

def get_diversity1(path):
    result=json.load(open(path, 'r'))
    print(len(result))
    gen=[]
    for dial in result.values():
        for turn in dial['log']:
            gen.append(turn['user'])
    print(get_richness(gen))

def get_lex_resp(path):
    result=json.load(open(path, 'r'))
    for dial in result:
        turn_domain=[]
        pv_b=None
        for turn in dial:
            bspn=turn['bspn']
            cons=reader.bspan_to_constraint_dict(bspn)
            cur_domain=list(cons.keys())
            if cur_domain==[]:
                turn_domain=['general']
            else:
                if len(cur_domain)==1:
                    turn_domain=cur_domain
                else:
                    if pv_b is None: # In rare cases, there are more than one domain in the first turn
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                turn_domain=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(reader.bspan_to_constraint_dict(pv_b).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                turn_domain=[domain]
            pv_b=bspn
            turn['lex_resp']=lex_resp(turn['resp'], bspn, turn['aspn'], turn_domain)
    json.dump(result, open(path, 'w'), indent=2)
        
def lex_resp(resp, bspn, aspn, turn_domain):
    value_map={}
    restored = resp
    restored=restored.replace('<sos_r>','')
    restored=restored.replace('<eos_r>','')
    restored.strip()
    restored = restored.capitalize()
    restored = restored.replace(' -s', 's')
    restored = restored.replace(' -ly', 'ly')
    restored = restored.replace(' -er', 'er')
    constraint_dict=reader.bspan_to_constraint_dict(bspn)#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
    mat_ents = reader.db.get_match_num(constraint_dict, True)
    #print(mat_ents)
    #print(constraint_dict)
    if '[value_car]' in restored:
        restored = restored.replace('[value_car]','toyota')
        value_map['taxi']={}
        value_map['taxi']['car']='toyota'

    # restored.replace('[value_phone]', '830-430-6666')
    domain=[]
    for d in turn_domain:
        if d.startswith('['):
            domain.append(d[1:-1])
        else:
            domain.append(d)
    act_dict=reader.aspan_to_act_dict(aspn)
    if len(act_dict)==1:
        domain=list(act_dict.keys())

    if list(act_dict.keys())==['police']:
        if '[value_name]' in restored:
            restored=restored.replace('[value_name]', 'parkside police station')
        if '[value_address]' in restored:
            restored=restored.replace('[value_address]', 'parkside , cambridge')
        if '[value_phone]' in restored:
            restored=restored.replace('[value_phone]', '01223358966')
    if list(act_dict.keys())==['hospital']:
        if '[value_address]' in restored:
            restored=restored.replace('[value_address]', 'Hills Rd, Cambridge')
        if '[value_postcode]' in restored:
            restored=restored.replace('[value_postcode]', 'CB20QQ')
    for d in domain:
        constraint = constraint_dict.get(d,None)
        if d not in value_map:
            value_map[d]={}
        if constraint:
            if 'stay' in constraint and '[value_stay]' in restored:
                restored = restored.replace('[value_stay]', constraint['stay'])
                value_map[d]['stay']=constraint['stay']
            if 'day' in constraint and '[value_day]' in restored:
                restored = restored.replace('[value_day]', constraint['day'])
                value_map[d]['day']=constraint['day']
            if 'people' in constraint and '[value_people]' in restored:
                restored = restored.replace('[value_people]', constraint['people'])
                value_map[d]['people']=constraint['people']
            if 'time' in constraint and '[value_time]' in restored:
                restored = restored.replace('[value_time]', constraint['time'])
                value_map[d]['time']=constraint['time']
            if 'type' in constraint and '[value_type]' in restored:
                restored = restored.replace('[value_type]', constraint['type'])
                value_map[d]['type']=constraint['type']
            if d in mat_ents and len(mat_ents[d])==0:
                for s in constraint:
                    if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                        restored = restored.replace('[value_price]', constraint['pricerange'])
                        value_map[d]['price']=constraint['pricerange']
                    if s+']' in restored:
                        restored = restored.replace('[value_%s]'%s, constraint[s])
                        value_map[d][s]=constraint[s]

        if '[value_choice' in restored and mat_ents.get(d):
            restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
            value_map[d]['choice']=str(len(mat_ents[d]))
    if '[value_choice' in restored:
        restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


    ent = mat_ents.get(domain[-1], [])
    d=domain[-1]
    if d not in value_map:
        value_map[d]={}
    if ent:
        # handle multiple [value_xxx] tokens first
        restored_split = restored.split()
        token_count = Counter(restored_split)
        for idx, t in enumerate(restored_split):
            if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                id1=t.index('_')
                id2=t.index(']')
                slot = t[id1+1:id2]
                pattern = r'\['+t[1:-1]+r'\]'
                for e in ent:
                    if e.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = e[slot].upper()
                        else:
                            rep = e[slot]
                        restored = re.sub(pattern, rep, restored, 1)
                        value_map[d][slot]=rep
                    elif slot == 'price' and  e.get('pricerange'):
                        restored = re.sub(pattern, e['pricerange'], restored, 1)
                        value_map[d][slot]=e['pricerange']

        # handle normal 1 entity case
        ent = ent[0]
        ents_list=reader.db.dbs[domain[-1]]
        ref_no=ents_list.index(ent)
        if ref_no>9:
            if '[value_reference]' in restored:
                restored = restored.replace('[value_reference]', '000000'+str(ref_no))
                value_map[d]['reference']='000000'+str(ref_no)
        else:
            if '[value_reference]' in restored:
                restored = restored.replace('[value_reference]', '0000000'+str(ref_no))
                value_map[d]['reference']='0000000'+str(ref_no)
        for t in restored.split():
            if '[value' in t:
                id1=t.index('_')
                id2=t.index(']')
                slot = t[id1+1:id2]
                if ent.get(slot):
                    if domain[-1] == 'hotel' and slot == 'price':
                        slot = 'pricerange'
                    if slot in ['name', 'address']:
                        rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                    elif slot in ['id','postcode']:
                        rep = ent[slot].upper()
                    else:
                        rep = ent[slot]
                    # rep = ent[slot]
                    rep='free' if slot in ['price', 'pricerange'] and rep=='?' else rep
                    if 'total fee' in restored and 'pounds' in rep:
                        price=float(rep.strip('pounds').strip())
                        people=constraint_dict[d].get('people', '1')
                        people=int(people) if people.isdigit() else 1
                        #calculate the total fee, people*price
                        rep = str(round(people*price, 2))+' pounds'
                    restored = restored.replace(t, rep)
                    value_map[d][slot]=rep
                    # restored = restored.replace(t, ent[slot])
                elif slot == 'price' and  ent.get('pricerange'):
                    rep='free' if ent['pricerange']=='?' else ent['pricerange']
                    restored = restored.replace(t, rep)
                    value_map[d][slot]=rep
                    # else:
                    #     print(restored, domain)       
    #restored = restored.replace('[value_area]', 'centre')
    for t in restored.split():
        if '[value' in t:
            slot=t[7:-1]
            value='UNKNOWN'
            for domain, sv in constraint_dict.items():
                if isinstance(sv, dict) and slot in sv:
                    value=sv[slot]
                    break
            if value=='UNKNOWN':
                for domain in mat_ents:
                    if len(mat_ents[domain])==0:
                        continue
                    ent=mat_ents[domain][0]
                    if slot in ent:
                        if slot in ['name', 'address']:
                            value=' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id', 'postcode']:
                            value=ent[slot].upper()
                        else:
                            value=ent[slot]
                        break
            if value!='UNKNOWN':
                if isinstance(value, str):
                    restored = restored.replace(t, value)
            else:
                for domain in constraint_dict.keys():
                    temp_ent=reader.db.dbs[domain][0]
                    if temp_ent.get(slot, None):
                        value=temp_ent[slot]
                        if isinstance(value, str):
                            restored = restored.replace(t, value)
                            break
    restored = restored.replace('[value_phone]', '01223462354')
    restored = restored.replace('[value_postcode]', 'cb21ab')
    restored = restored.replace('[value_address]', 'regent street')
    restored = restored.replace('[value_people]', 'several')
    restored = restored.replace('[value_day]', 'Saturday')
    restored = restored.replace('[value_time]', '12:00')
    restored = restored.split()
    for idx, w in enumerate(restored):
        if idx>0 and restored[idx-1] in ['.', '?', '!']:
            restored[idx]= restored[idx].capitalize()
    restored = ' '.join(restored)

    return restored.strip()

def collect_test_dials(path1, path2, path3, dial_num=25, groups=4):
    # path1: DS-GUS with GUS
    # path2: DS-SL with GUS
    # path3: DS-ABUS with ABUS
    result1=json.load(open(path1, 'r'))
    result2=json.load(open(path2, 'r'))
    result3=json.load(open(path3, 'r'))
    result3=[dial['log'] for dial in result3.values()]
    goal_list=json.load(open('analysis/goal_list.json', 'r'))
    zip_file=list(zip(result1, result2, result3, goal_list))
    zip_file=random.sample(zip_file, dial_num*groups)
    score={
            'Success':'',
            'Coherency-DS':'',
            'Fluency-DS':'',
            'Coherency-US':'',
            'Fluency-US':'',
            'Human-like':''
            }
    collected=[[] for _ in range(groups)]
    for i, (dial1, dial2, dial3, goal) in enumerate(zip_file):
        entry1={'goal':goal, 'log':simplify_dial(dial1), 'score':score}
        entry2={'goal':goal, 'log':simplify_dial(dial2), 'score':score}
        entry3={'goal':goal, 'log':simplify_dial(dial3), 'score':score}
        group_id=i//dial_num
        collected[group_id]+=[entry1, entry2, entry3]
        
    for id, col in enumerate(collected):
        json.dump(col, open('analysis/dials_for_test{}.json'.format(id), 'w'), indent=2)
        print('Group id:{}, dial nums:{}'.format(id, len(col)))

def collect_test_goals():
    goals=[]
    for i in range(4):
        path='analysis/dials_for_test{}.json'.format(i)
        data=json.load(open(path, 'r'))
        goals+=[dial['goal'] for i, dial in enumerate(data) if i%3==0]
    print('Goals for human eval:', len(goals))
    json.dump(goals, open('analysis/goals_for_test.json', 'w'), indent=2)

def temp1():
    dials=[]
    for i in range(4):
        dials.append(json.load(open('analysis/dials_for_test{}.json'.format(i))))
    new_res=json.load(open('/mnt/workspace/liuhong/RL_exp/RL-synthetic-reward-6-20-seed345/best_DS/validate_result1.json', 'r'))
    goals=json.load(open('analysis/goals_for_test.json', 'r'))
    print('Dial num:', len(new_res))
    score={
            'Success':'',
            'Coherency-DS':'',
            'Fluency-DS':'',
            'Coherency-US':'',
            'Fluency-US':'',
            'Human-like':''
            }
    for i, dial in enumerate(new_res):
        group_id=i//25
        dial_id=(i%25)*3
        entry={'goal':goals[i], 'log':simplify_dial(dial), 'score':score}
        dials[group_id][dial_id]=entry
    
    for id, col in enumerate(dials):
        json.dump(col, open('analysis/dials_for_human_test{}.json'.format(id), 'w'), indent=2)
        print('Group id:{}, dial nums:{}'.format(id, len(col)))

def simplify_dial(dial):
    new_dial=[]
    for turn in dial:
        new_turn={}
        new_turn['user']=turn['user'].replace('<sos_u>', '').replace('<eos_u>', '').strip()
        new_turn['resp']=turn['lex_resp']
        new_dial.append(new_turn)
    return new_dial

def add_metric(path):
    data=json.load(open(path,'r'))
    for dial in data:
        dial['score'].update({'Human like':''})
    json.dump(data, open(path, 'w'), indent=2)

def count_average_turn():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r'))
    count=0
    for dial_id, dial in data.items():
        count+=len(dial['log'])
    print('Average turn:', count/len(data))

if __name__=='__main__':
    #count_average_turn()
    #extract_goal_from_ABUS(path='analysis/gen_dials_1000.json')
    #evaluate_dialog_with_ABUS(path='/home/liuhong/myworkspace/analysis/RL-synthetic-reward-6-20_1000.json')
    #evaluate_dialog_with_ABUS(path='/home/liuhong/myworkspace/analysis/RL-synthetic-reward-6-20-seed345_1000.json')
    #evaluate_dialog_with_ABUS(path='/home/liuhong/myworkspace/analysis/RL-synthetic-reward-6-20-seed7474_1000.json')
    #evaluate_dialog_with_ABUS(path='/home/liuhong/myworkspace/analysis/RL-ABUS-seed7474_1000.json')
    #get_diversity('/mnt/workspace/liuhong/RL_exp/RL-4-24-gen-goal-only-ds/best_DS/validate_result.json')
    #get_diversity1('/home/liuhong/myworkspace/analysis/RL-ABUS-seed345_1000.json')
    #get_lex_resp('/mnt/workspace/liuhong/RL_exp/RL-synthetic-reward-6-20-seed345/best_DS/validate_result1.json')
    #collect_test_dials('/mnt/workspace/liuhong/RL_exp/RL-4-24-gen-goal-only-ds/best_DS/validate_result.json',\
     #   'experiments_21/turn-level-DS-97_34-otl/best_score_model/validate_result.json', 'analysis/RL-ABUS-seed345_1000.json', dial_num=25)
    #add_metric('analysis/collected1_50.json')
    #add_metric('analysis/collected2_50.json')
    #collect_test_goals()
    #collect_improvement()
    #temp()
    #get_attentions1('experiments_21/turn-level-DS-97_34-otl/best_score_model', encode_key=['bspn', 'resp'], mode='bspn', turn_th=4)
    #for n in range(4, 8):
     #   attentions=get_attentions('experiments_21/UBAR_seed11/best_score_model', turn_th=n)
      #  plot_ubar_attention_bar(attentions, n)
    #find_attention_case('experiments_21/UBAR_seed11/best_score_model', encode_key=['user', 'bspn', 'db', 'aspn', 'resp'], turn_th=4)
    #plot_attention_bar()
    #extract_goal()
    #path1='experiments_21/all_BRU_sd123_lr0.0001_bs8_ga4/best_score_model/result.json'
    #path2='experiments_21/SimpleTOD2/best_score_model/result.json'
    #path3='experiments_21/UBAR_seed11/best_score_model/result.json'
    #match_list1, success_list1, bleu_list1, dial_id_list=get_metrics_list(path1)
    #match_list2, success_list2, bleu_list2, _=get_metrics_list(path2, dial_order=dial_id_list)
    #match_list3, success_list3, bleu_list3, _=get_metrics_list(path3, dial_order=dial_id_list)
    #print([(b1, b2) for b1, b2 in zip(bleu_list1, bleu_list2)])
    #print('UBAR vs MGA:', McNemar(match_list1, match_list3), McNemar(success_list1, success_list3), matched_pair(bleu_list1, bleu_list3))
    #print('SimpleTOD vs MGA:', McNemar(match_list1, match_list2), McNemar(success_list1, success_list2), matched_pair(bleu_list1, bleu_list2))
    #print([(dial_id, s1, s2) for dial_id, s1, s2 in zip(dial_id_list, success_list1, success_list2) if s1==0 and s2==1])
    #find_unseen_usr_act(path1, path2)
    '''
    match_list, success_list1, bleu_list, dial_id_list=get_metrics_list('experiments_21/turn-level-DS-97_34-otl/best_score_model/result.json')
    match_list, success_list2, bleu_list, dial_id_list=get_metrics_list('RL_exp/RL-3-17-iter/best_DS/result.json')
    pool1, pool2 = [], []
    for i, (s1, s2) in enumerate(zip(success_list1, success_list2)):
        if not s1 and s2:
            pool1.append(dial_id_list[i])
        if not s2 and s1:
            pool2.append(dial_id_list[i])
    print('not success in 1 but success in 2:', len(pool1), pool1)
    print('not success in 2 but success in 1:', len(pool2), pool2)
    '''
    bspn='[restaurant] food spanish area centre pricerange expensive'
    constraint_dict=reader.bspan_to_constraint_dict(bspn)#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
    mat_ents = reader.db.get_match_num(constraint_dict, True)
    print(mat_ents)
