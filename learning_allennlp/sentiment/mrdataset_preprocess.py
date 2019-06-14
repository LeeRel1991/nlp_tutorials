"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: utils.py
@time: 2019/6/10 20:30
"""
from pathlib import Path

import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import traceback


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = os.path.expanduser(data_folder[0])
    neg_file = os.path.expanduser(data_folder[1])
    vocab = defaultdict(float)

    def spare_file(filename, label_id=0):
        f =  open(filename, "rb")
        for id, line in enumerate(f):
            try:
                line = line.decode()
                rev = [line.strip()]
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                # TODO pad for cnn, rnn 无影响
                if len(orig_rev.split()) < 10:
                    continue
                words = set(orig_rev.split())
                for word in words:  vocab[word] += 1
                datum = {"y": label_id,
                         "text": orig_rev,
                         "num_words": len(orig_rev.split()),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)
            except:
                # 跳过utf8无法解析的特殊字符
                print("invalid char ", id)
        f.close()
    spare_file(pos_file, label_id=1)
    spare_file(neg_file, label_id=0)

    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def split_into_train_test(revs, save_path, cv_ind=0):
    total_num = len(revs)

    # Randomly shuffle data
    # np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(total_num))
    # revs_shufled = revs[shuffle_indices]

    test = [rev  for rev in revs if rev['split']==cv_ind]
    train = [rev for rev in revs if rev['split'] != cv_ind]
    import os, pickle

    print(f"train samples: {len(train)}, test samples: {len(test)}")

    for filename, data in zip(['train.pkl','test.pkl' ], [train, test]):
        full_filename  = os.path.expanduser(os.path.join(save_path, filename))
        with open(full_filename, 'wb') as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    # w2v_file = sys.argv[1]
    data_folder = [Path("~/nlp/dataset/MRv1.0/rt-polaritydata")/x for x in ["rt-polarity.pos", "rt-polarity.neg"]]
    import os
    print("loading data...", os.path.abspath(data_folder[0]))
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)

    split_into_train_test(revs, '~/nlp/dataset/MRv1.0/rt-polaritydata', cv_ind=0)

    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))


