
import os
from functools import partial
import torch
import os.path
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import pdb
import config.cfg as cfg
import json
from tqdm import tqdm
from dataset.CNN_RNN import RNN_ENCODER
from torch.utils.data import Dataset
import glob
import pickle
import random
import pandas as pd
import numpy as np

args = cfg.parse_args()

import matplotlib.pyplot as plt

def get_imgs(img_path, transform=None,bbox=None):
    img = Image.open(img_path).convert('RGB')
    # plt.imshow(img)
    # plt.show()
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.65)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        # plt.imshow(img)
        # plt.show()

    #if transform is not None:
    img = transform(img)
    return img




class BIRDS(Dataset):
    def __init__(self, data_dir, split='train',imsize=64, transform=None):

        self.transform = transform
        self.embeddings_num = args.CAPTIONS_PER_IMAGE
        self.split=split
        self.img_size=imsize
        self.data = []
        self.data_dir = data_dir
        self.filenames = self.load_filenames(self.data_dir, self.split)

        self.bbox = self.load_bbox()
        self.filepath = os.path.join(self.data_dir, "new_birds_embedding.pickle")
        # self.text_encoder_dict1 = \
        #     torch.load(os.path.join(data_dir, "birds_text_encoder200.pth"),
        #                map_location=lambda storage, loc: storage)
        self.captions, self.ixtoword,self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        # self.text_encoder = RNN_ENCODER(self.n_words, nhidden=256)
        # self.text_encoder.load_state_dict(self.text_encoder_dict1)
        # for p in self.text_encoder.parameters():
        #     p.requires_grad = False


        if args.Iscondtion:
            self.sent_emb = self.load_embedding()

    def load_embedding(self):

        if not os.path.isfile(self.filepath):
            self.get_emb(self.captions)
            print("save test embedding,please reboot the programmer")
            return
        else :
            with open(self.filepath, 'rb') as f:
                x = pickle.load(f,encoding='bytes')
                return x

    def get_emb(self, captions):
        sent_embs_list = []
        elem_list = []
        elem_len_list = []
        a = 0
        for elem in (captions):
            a = a + 1

            elem = np.asarray(elem).astype('int64')
            num_words = len(elem)
            caption_len = num_words
            x = np.zeros((args.WORDS_NUM, 1), dtype='int64')
            if num_words <= args.WORDS_NUM:
                x[:num_words, 0] = elem
            else:
                ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:args.WORDS_NUM]
                ix = np.sort(ix)
                x[:, 0] = elem[ix]
                caption_len = args.WORDS_NUM
            elem_len_list.append(caption_len)
            elem_list.append(x)
            if a % args.CAPTIONS_PER_IMAGE == 0:
                print(a)
                elem_len_list_tensor = torch.LongTensor(elem_len_list)
                elem_list_tensor = torch.LongTensor(elem_list)
                elem_len_list.clear()
                elem_list.clear()
                hidden = self.text_encoder.init_hidden(args.CAPTIONS_PER_IMAGE)
                sent_emb = self.text_encoder(elem_list_tensor, elem_len_list_tensor, hidden=hidden)
                sent_embs_list.append(sent_emb)

        with open(self.filepath, 'ab') as f:
            pickle.dump(sent_embs_list, f, protocol=2)
            f.close()
        return

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        #print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox




    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('bytes').split('\n')
                #captions = f.read().split('\n')
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
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
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

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
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
        return  captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames


    def __getitem__(self, index):

        data_dir = self.data_dir

        key = self.filenames[index]
        bbox = self.bbox[key]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.transform,bbox)

        if args.Iscondtion:
            sent_ix = random.randint(0, self.embeddings_num - 1)
            sent_emb = self.sent_emb[index][sent_ix]
            return imgs, sent_emb
        else:
            return imgs

    def __len__(self):
        return len(self.filenames)


class COCO(Dataset):
    def __init__(self, data_dir, split='train',imsize=64, transform=None):

        self.transform = transform
        self.embeddings_num = args.CAPTIONS_PER_IMAGE
        self.split = split
        self.img_size = imsize
        self.data = []
        self.data_dir = data_dir
        self.images_path=os.path.join(data_dir,self.split)

        self.filenames = self.load_filenames(self.data_dir, self.split)

        # self.test_files=os.path.join(data_dir,'test_images')
        #self.build_test_images(self.data_dir,self.filenames,self.test_files)
        # self.text_encoder_dict1 = \
        #     torch.load(os.path.join(data_dir, "text_encoder100.pth"),
        #                map_location=lambda storage, loc: storage)
        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        # self.text_encoder = RNN_ENCODER(self.n_words, nhidden=256)
        # self.text_encoder.load_state_dict(self.text_encoder_dict1)
        # for p in self.text_encoder.parameters():
        #     p.requires_grad = False
        self.filepath = os.path.join(self.images_path, "coco_embedding_new.pickle")
        if not os.path.exists(self.filepath):
            self.get_emb(self.captions)


        if args.Iscondtion:
            self.sent_emb = self.load_embedding(self.filepath)

    def load_embedding(self,filepath):

        if not os.path.isfile(filepath):
            self.get_emb(self.captions)
            print("save test embedding,please reboot the programmer")
            return
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f, encoding='bytes')

            return x

    def get_imgs(self,img_path, transform=None):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        return img

    def get_emb(self, captions):
        sent_embs_list = []
        elem_list = []
        elem_len_list = []
        a = 0
        for elem in (captions):
            a = a + 1

            elem = np.asarray(elem).astype('int64')
            num_words = len(elem)
            caption_len = num_words
            x = np.zeros((args.WORDS_NUM, 1), dtype='int64')
            if num_words <= args.WORDS_NUM:
                x[:num_words, 0] = elem
            else:
                ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:args.WORDS_NUM]
                ix = np.sort(ix)
                x[:, 0] = elem[ix]
                caption_len = args.WORDS_NUM
            elem_len_list.append(caption_len)
            elem_list.append(x)
            if a % args.CAPTIONS_PER_IMAGE == 0:
                print(a)
                elem_len_list_tensor = torch.LongTensor(elem_len_list)
                elem_list_tensor = torch.LongTensor(elem_list)
                elem_len_list.clear()
                elem_list.clear()
                hidden = self.text_encoder.init_hidden(args.CAPTIONS_PER_IMAGE)
                sent_emb = self.text_encoder(elem_list_tensor, elem_len_list_tensor, hidden=hidden)
                sent_embs_list.append(sent_emb)

        with open(self.filepath, 'ab') as f:
            pickle.dump(sent_embs_list, f, protocol=2)
            f.close()
        return

    def load_captions(self, data_dir, filenames):
        all_captions = []
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
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
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

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)
            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
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

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)

            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def build_test_images(self,data_dir,filename,file):
        for i in range(len(filename)):
            img_name = '%s/images/%s.jpg' % (data_dir, filename[i])
            img = Image.open(img_name)
            img.save(os.path.join(file,'%s.jpg'%filename[i]))



    def __getitem__(self, index):

        key = self.filenames[index]
        img_name = '%s/images/%s.jpg' % (self.images_path, key)
        imgs = self.get_imgs(img_name,self.transform)
        if args.Iscondtion:
            sent_ix =  random.randint(0, self.embeddings_num - 1)
            sent_emb = self.sent_emb[index][sent_ix]
            return imgs, sent_emb
        else:
            return imgs


    def __len__(self):
        return len(self.filenames)





class Flowers(Dataset):
    def __init__(self, data_dir, split='train',imsize=64, transform=None):

        self.transform = transform
        self.embeddings_num = args.CAPTIONS_PER_IMAGE
        self.split = split
        self.img_size = imsize
        self.data = []
        self.data_dir = data_dir
        self.images_path=os.path.join(data_dir,self.split)

        self.filenames = self.load_filenames(self.data_dir, self.split)

        # self.test_files=os.path.join(data_dir,'test_images')
        #self.build_test_images(self.data_dir,self.filenames,self.test_files)
        # self.text_encoder_dict1 = \
        #     torch.load(os.path.join(data_dir, "text_encoder200.pth"),
        #                map_location=lambda storage, loc: storage)
        #self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        #self.text_encoder = RNN_ENCODER(self.n_words, nhidden=256)
        # self.text_encoder.load_state_dict(self.text_encoder_dict1)
        # for p in self.text_encoder.parameters():
        #     p.requires_grad = False

        #self.filepath = os.path.join(self.images_path, "lm_sje_flowers_c10_hybrid_0.00070_1_10_trainvalids.txt_iter16400.t7")
        # if args.Iscondtion:
        #     self.sent_emb = self.load_embedding(self.filepath)
        self.number_example = len(self.filenames)

    def load_embedding(self,filepath):

        if not os.path.isfile(filepath):
            #self.get_emb(self.captions)
            print("save test embedding,please reboot the programmer")
            return
        else:
            pickle.load=partial(pickle.load,encoding='latin1')
            pickle.Unpickler=partial(pickle.Unpickler, encoding="latin1")
            x=torch.load(filepath,map_location=lambda storage,loc:storage,pickle_module=pickle)
            return x

    # def get_emb(self, captions):
    #     sent_embs_list = []
    #     elem_list = []
    #     elem_len_list = []
    #     a = 0
    #     for elem in (captions):
    #         a = a + 1
    #
    #         elem = np.asarray(elem).astype('int64')
    #         num_words = len(elem)
    #         caption_len = num_words
    #         x = np.zeros((args.WORDS_NUM, 1), dtype='int64')
    #         if num_words <= args.WORDS_NUM:
    #             x[:num_words, 0] = elem
    #         else:
    #             ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
    #             np.random.shuffle(ix)
    #             ix = ix[:args.WORDS_NUM]
    #             ix = np.sort(ix)
    #             x[:, 0] = elem[ix]
    #             caption_len = args.WORDS_NUM
    #         elem_len_list.append(caption_len)
    #         elem_list.append(x)
    #         if a % 10 == 0:
    #             print(a)
    #             elem_len_list_tensor = torch.LongTensor(elem_len_list)
    #             elem_list_tensor = torch.LongTensor(elem_list)
    #             elem_len_list.clear()
    #             elem_list.clear()
    #             hidden = self.text_encoder.init_hidden(10)
    #             sent_emb = self.text_encoder(elem_list_tensor, elem_len_list_tensor, hidden=hidden)
    #             sent_embs_list.append(sent_emb)
    #     with open(self.filepath, 'ab') as f:
    #         pickle.dump(sent_embs_list, f, protocol=2)
    #         f.close()
    #     return

    def load_captions(self, data_dir, filenames):
        all_captions = []
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
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
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

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)
            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
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

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        path=os.path.join(data_dir,split)

        filenames = []
        filelist=os.listdir(path+'/images')
        for filename in filelist:
            filenames.append(filename.split(".jpg")[0])
        return filenames

    def build_test_images(self,data_dir,filename,file):
        for i in range(len(filename)):
            img_name = '%s/images/%s.jpg' % (data_dir, filename[i])
            img = Image.open(img_name)
            img.save(os.path.join(file,'%s.jpg'%filename[i]))




    def __getitem__(self, index):

        key = self.filenames[index]
        img_name = '%s/images/%s.jpg' % (self.images_path, key)
        imgs = get_imgs(img_name, self.transform)
        if args.Iscondtion:
            sent_ix = random.randint(0, self.embeddings_num - 1)
            sent_emb2 = self.sent_emb[index][sent_ix]
            return imgs, sent_emb2
        else:
            return imgs

    def __len__(self):
        return len(self.filenames)



class CeleBA(Dataset):

    def __init__(self, root,transform=None,split='train'):

        self.data_dir=os.path.join(root,split)
        self.image_path=os.path.join(self.data_dir,'images')
        self.filenames=os.listdir(self.image_path)
        self.transform=transform


    def __getitem__(self, idx):

        data_dir = self.data_dir
        key = self.filenames[idx]
        img_name = '%s/images/%s' % (data_dir, key)
        imgs = get_imgs(img_name, self.transform)

        return imgs

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.filenames)