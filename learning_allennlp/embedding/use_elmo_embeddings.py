"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: use_elmo.py
@time: 2019/4/28 15:08
"""
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data import Vocabulary

from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.modules.elmo import batch_to_ids, Elmo
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

from learning_allennlp.dataset import JigsawDatasetReader
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)


if __name__ == "__main__":

    # pretrained elmo LM model, transformed from bilm-tf with dump_weights in bin/training.py
    options_file = './models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = './models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    # -------------------- first usage ---------------------------
    # 与Elmo类相比，增加了project 功能，即通过linear 将elmo的embedding 的dim转为指定的projection_dim
    # 默认指定elmo中的 num_output_representations=1如果参数1改为2，则表示产生两个权重不同的ELMo词向量，可以用于不同任务的词向量。
    # see https://blog.htliu.cn/2019/01/01/elmo-1/
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    # word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    # print(word_embeddings.get_output_dim())
    #
    # # use batch_to_ids to convert sentences to character ids
    # sentence_lists = [["I have a dog"], ["How are you , today is Monday"], ["I am fine thanks"]]
    # character_ids = batch_to_ids(sentence_lists)
    # embeddings = elmo_embedder(character_ids)
    # print(character_ids.shape, embeddings.shape)


    # -------------------- another usage ---------------------
    # from allennlp.commands.elmo import ElmoEmbedder
    elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)
    tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
    vectors = elmo.embed_sentence(tokens)
    print(vectors.shape)

    # calculate distance based on elmo embedding
    import scipy
    tokens2 = ["I", "ate", "an", "carrot", "for", "breakfast"]
    vectors2 = elmo.embed_sentence(tokens2)
    for layer_id in range(len(vectors)):
        print('\nvector ', vectors[layer_id][3], vectors[layer_id][3])
        distance = scipy.spatial.distance.cosine(vectors[layer_id][3], vectors2[layer_id][3])
        print(f"layer {layer_id} embedding distance: {distance}")



    # ------------- direct usgae

    # elmo2 = Elmo(options_file=options_file,
    #             weight_file=weight_file,
    #             num_output_representations=1)
    #
    # sentence_lists = [["I", 'have', 'a', "dog"], ["How", 'are' ,'you', ',', 'today', 'is', "Monday"]]
    # character_ids = batch_to_ids(sentence_lists)
    # embeddings2 = elmo2(character_ids)
    # print("elmo ", embeddings2['elmo_representations'][0].shape)

    # predictor
    # from allennlp.predictors.predictor import Predictor
    # predictor = Predictor.from_path()