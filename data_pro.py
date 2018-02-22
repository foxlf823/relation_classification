# -*- coding: utf-8 -*-
import numpy as np
import logging
from collections import Counter
import statistics
from collections import OrderedDict



def relationID2Name(id):
    if id == 0:
        return "Component-Whole(e2,e1)"
    elif id == 1:
        return "Other"
    elif id == 2:
        return "Instrument-Agency(e2,e1)"
    elif id == 3:
        return "Member-Collection(e1,e2)"
    elif id == 4:
        return "Cause-Effect(e2,e1)"
    elif id == 5:
        return "Entity-Destination(e1,e2)"
    elif id == 6:
        return "Content-Container(e1,e2)"
    elif id == 7:
        return "Message-Topic(e1,e2)"
    elif id == 8:
        return "Product-Producer(e2,e1)"
    elif id == 9:
        return "Member-Collection(e2,e1)"
    elif id == 10:
        return "Entity-Origin(e1,e2)"
    elif id == 11:
        return "Cause-Effect(e1,e2)"
    elif id == 12:
        return "Component-Whole(e1,e2)"
    elif id == 13:
        return "Message-Topic(e2,e1)"
    elif id == 14:
        return "Product-Producer(e1,e2)"
    elif id == 15:
        return "Entity-Origin(e2,e1)"
    elif id == 16:
        return "Content-Container(e2,e1)"
    elif id == 17:
        return "Instrument-Agency(e1,e2)"
    elif id == 18:
        return "Entity-Destination(e2,e1)"
    else:
        logger.debug('unknown relation id {} !!!!!!!!'.format(id))
        return None;

# when_not_use_other, use this function to map other to the last id
def rawID2innerID(id):
    if id == 0:
        return 0
    elif id == 1:
        return 18
    else:
        return id-1
    
def innerID2rawID(id):
    if id == 0:
        return 0
    elif id == 18:
        return 1
    else:
        return id+1
    
def outputToSem10rc(ids, path, include_other):
    startSentID = 8001
    
    with open(path, 'w') as f:
        for id in ids:
            if include_other:
                f.write('{}\t{}\n'.format(startSentID, relationID2Name(id)))
            else:
                f.write('{}\t{}\n'.format(startSentID, relationID2Name(innerID2rawID(id))))
            
            startSentID += 1

ENG_PUNC = set(['`','~','!','@','#','$','%','&','*','(',')','-','_','+','=','{',
                '}','|','[',']','\\',':',';','\'','"','<','>',',','.','?','/'])

DIGIT = set(['0','1','2','3','4','5','6','7','8','9'])


def normalizeWordList(list, cased):
    '''
    Normalize a list of words.
    alphabet - lower case
    digit - 0
    punctuation - #
    '''
    newlist = []
    for word in list:
        newword = ''
        for ch in word:
#             if ch in DIGIT:
#                 newword = newword + '0'
#             elif ch in ENG_PUNC: 
#                 newword = newword + '#'
#             else:
                newword = newword + ch
        if not cased:
            newword = newword.lower()       
        newlist.append(newword)
    
    return newlist

def normalizeWord(word, cased):

    newword = ''
    for ch in word:
#         if ch in DIGIT:
#             newword = newword + '0'
#         elif ch in ENG_PUNC: 
#             newword = newword + '#'
#         else:
            newword = newword + ch
    
    if not cased:
        newword = newword.lower() 
    return newword


def load_data(file, cased, use_word_between, include_other):
    sentences = []
    relations = []
    e1_pos = []
    e2_pos = []
    
    max_len = 0
    len_counter = Counter()

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():

            line = line.strip().split()
            
            if include_other:
                relations.append(int(line[0]))
            else:
                relations.append(rawID2innerID(int(line[0])))

            word_list = line[5:]

            if use_word_between:
                begin = int(line[1])
                end = int(line[4])
                e1_pos.append((0, int(line[2])-begin))  # (start_pos, end_pos)
                e2_pos.append((int(line[3])-begin, end-begin))  # (start_pos, end_pos)
                sentences.append(normalizeWordList(word_list[begin:end+1], cased))
                sentence_length = end+1-begin
            else:
                e1_pos.append((int(line[1]), int(line[2])))  # (start_pos, end_pos)
                e2_pos.append((int(line[3]), int(line[4])))  # (start_pos, end_pos)
                sentences.append(normalizeWordList(word_list, cased))
                sentence_length = len(word_list)
            
            

            if sentence_length > max_len:
                max_len = sentence_length
                
            if sentence_length not in len_counter:
                len_counter[sentence_length] = 1
            else:
                len_counter[sentence_length] += 1
                
    print("max sentence length %d" % (max_len))
#     print(len_counter.most_common())

    return sentences, relations, e1_pos, e2_pos, max_len


# def build_dict(sentences):
#     word_count = Counter()
#     for sent in sentences:
#         for w in sent:
#             if w not in word_count:
#                 word_count[w] = 1
#             else:
#                 word_count[w] += 1
#     # 按照词频降序返回 [('棒', 2), ('的', 1), ('青', 1), ('年', 1)]
#     ls = word_count.most_common()
# 
#     word_dict = {w[0]: index + 1 for (index, w) in enumerate(ls)}
#     # leave 0 to PAD
#     return word_dict

def create_alphabet(sentences):
    word_alpha = set()
    for sent in sentences:
        for w in sent:
            word_alpha.add(w)
    return word_alpha

def build_fixed_dict(alpha):
    word_dict = OrderedDict()
    index = 1 # leave 0 to PAD
    for w in alpha:
        word_dict[w] = index
        index += 1
        
    return word_dict

# def build_fixed_dict(sentences):
#     word_dict = OrderedDict()
#     index = 1 # leave 0 to PAD
#     for sent in sentences:
#         for w in sent:
#             if w not in word_dict:
#                 word_dict[w] = index
#                 index += 1
# 
#     return word_dict

def build_position_dict(max_len):
    '''
    If max_len is 40, the position will be between [-39, 39].
    And we use 40 as the position of PAD.
    '''

    position_dict = {w: index for (index, w) in enumerate(range(-(max_len-1), max_len+1))}
    return position_dict

def numpyNormalize(Z):
    Zmax,Zmin = Z.max(),Z.min()
    Z = (Z - Zmin)/(Zmax - Zmin)
    return Z

def load_embedding_from_glove(emb_file, word_dict, cased):
    
    vocab = {}
    with open(emb_file, 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break;
            templist = line.strip().split()
            # norm the value into [0,1] and map into [-0.01, 0.01]
#             vector = np.array(list(map(float, templist[1:])))
#             vector = (numpyNormalize(vector)*0.02-0.01)
#             vector = vector/np.linalg.norm(vector)
#             vocab[templist[0]] = vector

            try:
                vocab[normalizeWord(templist[0], cased)] = list(map(float, templist[1:]))
            except ValueError:
                continue
            
#     print('##############', vocab['has'])
        
    dim = len(vocab['the']) # assume 'the' exists
#     num_words = len(word_dict) + 1
    num_words = len(word_dict) 
    embeddings = np.random.uniform(-0.01, 0.01, size=(num_words, dim))
#     embeddings = np.random.normal(0, 1, size=(num_words, dim))
    
    pre_trained = 0
    for w in vocab.keys():
        if w in word_dict:
            embeddings[word_dict[w]-1] = np.array(vocab[w])
            pre_trained += 1
#     embeddings[0] = np.zeros(dim)

    logging.info(
        'embedding dimension %d' % (dim))

    logging.info(
        'embeddings: %.2f%%(pre_trained) total: %d' % (pre_trained / num_words * 100, num_words))

    return embeddings.astype(np.float32)


def pos(x, max):
    '''
    map the relative distance between [0, max)
    max should be odd
    '''
    half = max//2 - 1
    if x < -half:
        return 0
    if x >= -half and x <= half:
        return x + half + 1
    if x > half:
        return max-1


def vectorize(data, word_dict, max_len, position_dict):
    sentences, relations, e1_pos, e2_pos, _ = data

    # replace word with word-id
    e1_vec = []
    e2_vec = []
    
    # compute relative distance
    dist1 = []
    dist2 = []
    POSTION_PAD_ID = len(position_dict)-1 # see build_position_dict

    num_data = len(sentences)
    sents_vec = np.zeros((num_data, max_len), dtype=int)

    logging.debug('data shape: (%d, %d)' % (num_data, max_len))

    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        
        position_e1 = [ idx-pos1[1]  for idx, _ in enumerate(sent)]
        vec_position_e1 = [position_dict[pos] for pos in position_e1]
        pad_vec_position_e1 = [POSTION_PAD_ID]*max_len # see build_position_dict
        
        position_e2 = [ idx-pos2[1] for idx, _ in enumerate(sent)]
        vec_position_e2 = [position_dict[pos] for pos in position_e2]
        pad_vec_position_e2 = [POSTION_PAD_ID]*max_len

        # pad in the rear, sent in the front
        if max_len >= len(vec):
            sents_vec[idx, 0:len(vec)] = vec
            pad_vec_position_e1[0:len(vec)] = vec_position_e1
            pad_vec_position_e2[0:len(vec)] = vec_position_e2
        else:
            sents_vec[idx, 0:max_len] = vec[0:max_len]
            pad_vec_position_e1[0:max_len] = vec_position_e1[0:max_len]
            pad_vec_position_e2[0:max_len] = vec_position_e2[0:max_len]
            
        # pad in the front, sent in the rear
#         if max_len >= len(vec):
#             sents_vec[idx, max_len-len(vec):] = vec
#             pad_vec_position_e1[max_len-len(vec):] = vec_position_e1
#             pad_vec_position_e2[max_len-len(vec):] = vec_position_e2
#         else:
#             sents_vec[idx, 0:max_len] = vec[0:max_len]
#             pad_vec_position_e1[0:max_len] = vec_position_e1[0:max_len]
#             pad_vec_position_e2[0:max_len] = vec_position_e2[0:max_len]

        # last word of e1 and e2
        e1_vec.append(vec[pos1[1]])
        e2_vec.append(vec[pos2[1]])

        dist1.append(pad_vec_position_e1)
        dist2.append(pad_vec_position_e2)

#     for sent, p1, p2 in zip(sents_vec, e1_pos, e2_pos):
#         # current word position - last word position of e1 or e2
#         dist1.append([pos(p1[1] - idx, max_len) for idx, _ in enumerate(sent)])
#         dist2.append([pos(p2[1] - idx, max_len) for idx, _ in enumerate(sent)])

    
    # sents_vec - map each word of sentence to their word id
    # relations - relation class of sentence
    # e1_vec - map last word of entity 1 to its word id
    # e2_vec - map last word of entity 1 to its word id
    # dist1 - map each word position of entity 1 to it pos id
    # dist2 - map each word position of entity 2 to it pos id 
    return sents_vec, relations, e1_vec, e2_vec, dist1, dist2
