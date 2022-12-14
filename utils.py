import logging
import json
import os
import numpy as np
from collections import OrderedDict
import transformers
import ontology
import torch

def prepare_for_std_eval(path=None, data=None):
    if path:
        data=json.load(open(path, 'r', encoding='utf-8'))
    new_data={}
    dials=pack_dial(data)
    for dial_id in dials:
        new_data[dial_id]=[]
        dial=dials[dial_id]
        for turn in dial:
            if turn['user']=='':
                continue
            entry={}
            entry['response']=turn['resp_gen']
            entry['state']=bspan_to_constraint_dict(turn['bspn_gen'])
            new_data[dial_id].append(entry)
    if path:
        new_path=path[:-5]+'std.json'
        json.dump(new_data, open(new_path, 'w'), indent=2)
    return new_data

def bspan_to_constraint_dict(bspan, bspn_mode='bspn'):
    bspan = bspan.split() if isinstance(bspan, str) else bspan
    constraint_dict = {}
    domain = None
    conslen = len(bspan)
    for idx, cons in enumerate(bspan):
        if cons == '<eos_b>':
            break
        if '[' in cons:
            if cons[1:-1] not in ontology.all_domains:
                continue
            domain = cons[1:-1]
        elif cons in ontology.get_slot:
            if domain is None:
                continue
            if cons == 'people':
                try:
                    ns = bspan[idx+1]
                    if ns == "'s":
                        continue
                except:
                    continue
            if not constraint_dict.get(domain):
                constraint_dict[domain] = {}
            if bspn_mode == 'bsdx':
                constraint_dict[domain][cons] = 1
                continue
            vidx = idx+1
            if vidx == conslen:
                break
            vt_collect = []
            vt = bspan[vidx]
            while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                vt_collect.append(vt)
                vidx += 1
                if vidx == conslen:
                    break
                vt = bspan[vidx]
            if vt_collect:
                constraint_dict[domain][cons] = ' '.join(vt_collect)

    return constraint_dict

def pack_dial(data):
    dials = {}
    for turn in data:
        dial_id = turn['dial_id']
        if dial_id not in dials:
            dials[dial_id] = []
        dials[dial_id].append(turn)
    return dials

def modified_encode(tokenizer, text):
    if int(transformers.__version__[0])>=3:
        if isinstance(text, str):
            word_list=text.split()
        elif isinstance(text, list):
            word_list=text
        else:             
            raise TypeError(text)
        special_token_pos=[]
        results=[]
        for idx, word in enumerate(word_list):
            if word in tokenizer.additional_special_tokens:
                special_token_pos.append(idx)
        for j, idx in enumerate(special_token_pos):
            if j<len(special_token_pos)-1:
                next_idx=special_token_pos[j+1]
                results+=tokenizer.encode(word_list[idx]) + tokenizer.encode(' '+' '.join(word_list[idx+1:next_idx]))
            else:
                results+=tokenizer.encode(word_list[idx])
                if idx<len(word_list)-1:# the last word is not a special token
                    results+=tokenizer.encode(' '+' '.join(word_list[idx+1:]))
        return results

    else:
        return tokenizer.encode(text)

def kl_loss(p_proba, q_proba): # [B, T, V] or [T,V]
    eps=1e-45
    dim=p_proba.dim()
    loss = q_proba * (torch.log(q_proba+eps) - torch.log(p_proba+eps))
    loss = torch.sum(loss, dim=-1)   # sum over vocabulary
    loss = torch.sum(loss, dim=-1)   # sum over sequence
    if dim==2:
        return loss
    else:
        return loss.mean()

def Loss1(p_proba, q_proba):
    eps=1e-45
    dim=p_proba.dim()
    loss=torch.log(q_proba+eps)-torch.log(p_proba+eps)
    loss=torch.sum(loss, dim=-1)
    loss=torch.sum(loss, dim=-1)
    if dim==2:
        return loss
    else:
        return loss.mean()

def modify_map_file(path):
    # modify special_tokens_map.json for different versions of GPT2Tokenizer
    file_path=os.path.join(path, 'special_tokens_map.json')
    map=json.load(open(file_path,'r', encoding='utf-8'))
    if transformers.__version__[:2]=='2.':
        for key in map:
            if isinstance(map[key],dict):
                map[key]=map[key]['content']
    json.dump(map, open(file_path, 'w'))

    file_path=os.path.join(path, 'tokenizer_config.json')
    config=json.load(open(file_path,'r', encoding='utf-8'))
    if transformers.__version__[:2]=='2.':
        config={}
    json.dump(config, open(file_path, 'w'))




def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()???????????????????????????top_k??????????????????????????????(values,indices)
        # ...??????????????????????????????????????????
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value  # ??????topk????????????????????????logits??????????????????

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True)  # ???logits??????????????????
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def py2np(list):
    return np.array(list)


def write_dict(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, indent=2)

def f1_score(label_list, pred_list):
    tp = len([t for t in pred_list if t in label_list])
    fp = max(0, len(pred_list) - tp)
    fn = max(0, len(label_list) - tp)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1

class Vocab(object):
    def __init__(self, vocab_size=0):
        self.vocab_size = vocab_size
        self.vocab_size_oov = 0   # get after construction
        self._idx2word = {}   #word + oov
        self._word2idx = {}   # word
        self._freq_dict = {}   #word + oov
        for w in ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>','<eos_u>', '<eos_r>',
                      '<eos_b>', '<eos_a>', '<go_d>','<eos_d>']:
            self._absolute_add_word(w)

    def _absolute_add_word(self, w):
        idx = len(self._idx2word)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: %d' % (len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_slots:
            self._add_to_vocab(word)
        for word in l:
            if word.startswith('[value_') and word.endswith(']'):
                self._add_to_vocab(word)
        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path+'.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_oov = len(self._idx2word)
        print('vocab file loaded from "'+vocab_path+'"')
        print('Vocabulary size including oov: %d' % (self.vocab_size_oov))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict(vocab_path+'.word2idx.json', self._word2idx)
        write_dict(vocab_path+'.freq.json', _freq_dict)


    def encode(self, word, include_oov=True):
        if include_oov:
            if self._word2idx.get(word, None) is None:
                raise ValueError('Unknown word: %s. Vocabulary should include oovs here.'%word)
            return self._word2idx[word]
        else:
            word = '<unk>' if word not in self._word2idx else word
            return self._word2idx[word]

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def oov_idx_map(self, idx):
        return 2 if idx > self.vocab_size else idx

    def sentence_oov_map(self, index_list):
        return [self.oov_idx_map(_) for _ in index_list]


    def decode(self, idx, indicate_oov=False):
        if not self._idx2word.get(idx):
            raise ValueError('Error idx: %d. Vocabulary should include oovs here.'%idx)
        if not indicate_oov or idx<self.vocab_size:
            return self._idx2word[idx]
        else:
            return self._idx2word[idx]+'(o)'

    def sentence_decode(self, index_list, eos=None, indicate_oov=False):
        l = [self.decode(_, indicate_oov) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]

def padSeqs_gpt(sequences, pad_id, maxlen=None):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_mexlen = np.max(lengths)

    # maxlen = 1024
    if seq_mexlen > 1024: # gpt2.n_ctx
        # print('maxlen exceeds 1024')
        maxlen = 1024
    else:
        maxlen = seq_mexlen

    # tokenizer.encode('<|endoftext|>') = ['50256']
    # All labels set to ``-100`` are ignored (masked), the loss is only
    # computed for labels in ``[0, ..., config.vocab_size]`` (from modeling_gpt2.GPT2LMHeadModel)
    
    x = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        # trunc method = 'pre'
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)

        # pad method = 'post'
        x[idx, :len(trunc)] = trunc
            
    return x, lengths


    
        


def padSeqs(sequences, maxlen=None, truncated = False, pad_method='post',
                     trunc_method='pre', dtype='int32', value=0.):
    if not hasattr(sequences, '__len__'): 
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    if maxlen is not None and truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list/array was found')
            continue  # empty list/array was found
        if trunc_method == 'pre':
            trunc = s[-maxlen:]
        elif trunc_method == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % trunc_method)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if pad_method == 'post':
            x[idx, :len(trunc)] = trunc
        elif pad_method == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % pad_method)
    return x


def get_glove_matrix(glove_path, vocab, initial_embedding_np):
    """
    return a glove embedding matrix
    :param self:
    :param glove_file:
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    ef = open(glove_path, 'r', encoding='UTF-8')
    cnt = 0
    vec_array = initial_embedding_np
    old_avg = np.average(vec_array)
    old_std = np.std(vec_array)
    vec_array = vec_array.astype(np.float32)
    new_avg, new_std = 0, 0

    for line in ef.readlines():
        line = line.strip().split(' ')
        word, vec = line[0], line[1:]
        vec = np.array(vec, np.float32)
        if not vocab.has_word(word):
            continue
        word_idx = vocab.encode(word)
        if word_idx <vocab.vocab_size:
            cnt += 1
            vec_array[word_idx] = vec
            new_avg += np.average(vec)
            new_std += np.std(vec)
    new_avg /= cnt
    new_std /= cnt
    ef.close()
    logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (cnt, old_avg,
                                                                                          new_avg, old_std, new_std))
    return vec_array


def position_encoding_init(self, n_position, d_pos_vec):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
                             if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc
