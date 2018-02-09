# -*- coding: utf-8 -*-
import numpy as np
import logging
from collections import Counter



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
    
def outputToSem10rc(ids, path):
    startSentID = 8001
    
    with open(path, 'w') as f:
        for id in ids:
            f.write('{}\t{}\n'.format(startSentID, relationID2Name(id)))
            
            startSentID += 1

ENG_PUNC = set(['`','~','!','@','#','$','%','&','*','(',')','-','_','+','=','{',
                '}','|','[',']','\\',':',';','\'','"','<','>',',','.','?','/'])

DIGIT = set(['0','1','2','3','4','5','6','7','8','9'])


def normalizeWordList(list):
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
                
        newlist.append(newword.lower())
    
    return newlist

def normalizeWord(word):

    newword = ''
    for ch in word:
#         if ch in DIGIT:
#             newword = newword + '0'
#         elif ch in ENG_PUNC: 
#             newword = newword + '#'
#         else:
            newword = newword + ch
    
    return newword.lower()


def load_data(file):
    sentences = []
    relations = []
    e1_pos = []
    e2_pos = []
    
    max_len = 0

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():

            line = line.strip().split()
            relations.append(int(line[0]))
            e1_pos.append((int(line[1]), int(line[2])))  # (start_pos, end_pos)
            e2_pos.append((int(line[3]), int(line[4])))  # (start_pos, end_pos)

            sentences.append(normalizeWordList(line[5:]))

            if len(line[5:]) > max_len:
                max_len = len(line[5:])
                
    print("max sentence length %d" % (max_len))

    return sentences, relations, e1_pos, e2_pos


def build_dict(sentences):
    word_count = Counter()
    for sent in sentences:
        for w in sent:
            if w not in word_count:
                word_count[w] = 1
            else:
                word_count[w] += 1
    # 按照词频降序返回 [('棒', 2), ('的', 1), ('青', 1), ('年', 1)]
    ls = word_count.most_common()

    word_dict = {w[0]: index + 1 for (index, w) in enumerate(ls)}
    # leave 0 to PAD
    return word_dict

def numpyNormalize(Z):
    Zmax,Zmin = Z.max(),Z.min()
    Z = (Z - Zmin)/(Zmax - Zmin)
    return Z

def load_embedding_from_glove(emb_file, word_dict):
    
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

            vocab[normalizeWord(templist[0])] = list(map(float, templist[1:]))

        
    dim = len(vocab['the']) # assume 'the' exists
    num_words = len(word_dict) + 1
    embeddings = np.random.uniform(-0.01, 0.01, size=(num_words, dim))
    
    pre_trained = 0
    for w in vocab.keys():
        if w in word_dict:
            embeddings[word_dict[w]] = np.array(vocab[w])
            pre_trained += 1
    embeddings[0] = np.zeros(dim)

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


def vectorize(data, word_dict, max_len):
    sentences, relations, e1_pos, e2_pos = data

    # replace word with word-id
    e1_vec = []
    e2_vec = []

    num_data = len(sentences)
    sents_vec = np.zeros((num_data, max_len), dtype=int)

    logging.debug('data shape: (%d, %d)' % (num_data, max_len))

    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]

        # padding to front
#         if max_len >= len(vec):
#             sents_vec[idx, 0:len(vec)] = vec
#         else:
#             sents_vec[idx, 0:max_len] = vec[0:max_len]
            
        # padding to rear
        if max_len >= len(vec):
            sents_vec[idx, max_len-len(vec):] = vec
        else:
            sents_vec[idx, 0:max_len] = vec[0:max_len]

        # last word of e1 and e2
        e1_vec.append(vec[pos1[1]])
        e2_vec.append(vec[pos2[1]])

    # compute relative distance
    dist1 = []
    dist2 = []

    for sent, p1, p2 in zip(sents_vec, e1_pos, e2_pos):
        # current word position - last word position of e1 or e2
        dist1.append([pos(p1[1] - idx, max_len) for idx, _ in enumerate(sent)])
        dist2.append([pos(p2[1] - idx, max_len) for idx, _ in enumerate(sent)])
    
    # sents_vec - map each word of sentence to their word id
    # relations - relation class of sentence
    # e1_vec - map last word of entity 1 to its word id
    # e2_vec - map last word of entity 1 to its word id
    # dist1 - map each word position of entity 1 to it pos id
    # dist2 - map each word position of entity 2 to it pos id 
    return sents_vec, relations, e1_vec, e2_vec, dist1, dist2
