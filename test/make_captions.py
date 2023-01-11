import numpy as np
import torch
import os.path
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import pickle


def load_embedding( filepath,index):
    pick_list=[]
    with open(filepath, 'rb') as f:
        x = pickle.load(f, encoding='bytes')
        with open('gen.pickle', 'ab') as f2:
            for elem in index:
                a=x[elem]
                pick_list.append(a)
            pickle.dump(pick_list, f2, protocol=2)
            f2.close()
        f.close()
    #return x
def build_dictionary(train_captions, test_captions):
    word_counts = defaultdict(float)
    captions = train_captions + test_captions
    for sent in captions:
        for word in sent:
            word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    train_captions_new = []
    for t in train_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        train_captions_new.append(rev)

    test_captions_new = []
    for t in test_captions:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        test_captions_new.append(rev)

    return [train_captions_new, test_captions_new,
            ixtoword, wordtoix, len(ixtoword)]

def load_filenames(data_dir, split):
    filepath = '%s/%s/filenames.pickle' % (data_dir, split)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)

        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    else:
        filenames = []
    return filenames
def load_captions(data_dir, filenames):
    all_captions = []
    embeddings_num=18
    for i in range(len(filenames)):
        cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
        with open(cap_path, "r") as f:
            captions = f.read().decode('bytes').split('\n')
            # captions = f.read().split('\n')
            cnt = 0
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
                cnt += 1
                if cnt == embeddings_num:
                    break
            if cnt < embeddings_num:
                print('ERROR: the captions for %s less than %d'
                      % (filenames[i], cnt))
    return all_captions

def load_text_data(data_dir, split):
    filepath = os.path.join(data_dir, 'captions.pickle')
    train_names = load_filenames(data_dir, 'train')
    test_names = load_filenames(data_dir, 'test')
    if not os.path.isfile(filepath):
        train_captions = load_captions(data_dir, train_names)
        test_captions = load_captions(data_dir, test_names)
        train_captions, test_captions, ixtoword, wordtoix, n_words = \
            build_dictionary(train_captions, test_captions)
        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, test_captions,
                         ixtoword, wordtoix], f, protocol=2)
            print('Save to: ', filepath)
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', filepath)
    if split == 'train':
        # a list of list: each list contains
        # the indices of words in a sentence
        captions = train_captions
    else:  # split=='test'
        captions = test_captions
    return captions, ixtoword, wordtoix, n_words

def load_txt(source_path):
    with open("all_.txt","a")as f1:
        file_name=os.listdir(source_path)
        for i in range(len(file_name)):
            file_path=os.path.join(source_path,file_name[i])
            with open(file_path,'r') as f2:
                txt=str(f2.readlines())
                f1.write(file_name[i])
                f1.write('\n')
                f1.write(txt)
                f1.write('\n')
                f2.close()



if __name__=="__main__":
    index=[23955, 27056,8537,2960,1398,22595,439,37734,20086,24943,31040,9919]

    text_dir="F:/datasets/coco/val2014"
    data_dir='F:/datasets/coco/test'
    filepath = os.path.join(data_dir, "coco_embedding_new.pickle")
    filepath2='F:/datasets/coco_2/coco/train/char-CNN-RNN-embeddings.pickle'
    load_embedding(filepath2,index)

    #load_text_data(data_dir, 'test')
    #load_txt(text_dir)