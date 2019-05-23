"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: sst_lstm.py
@time: 2019/5/20 11:57
"""
from pathlib import Path

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer
from allennlp.nn import util as nn_util
from allennlp.nn.util import get_text_field_mask

from learning_allennlp.sentiment.sst_classifier import SSTClassifier


def train_main():
    train_reader = StanfordSentimentTreeBankDatasetReader(use_subtrees=True)
    dev_reader = StanfordSentimentTreeBankDatasetReader(use_subtrees=False)
    train_dataset = train_reader.read('~/nlp/dataset/sst/trees/train.txt')
    dev_dataset = dev_reader.read('~/nlp/dataset/sst/trees/dev.txt')

    print(f"total train samples: {len(train_dataset)}, dev samples: {len(dev_dataset)}")

    # 建立词汇表，从数据集中建立
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)
    vocab_dim = vocab.get_vocab_size('tokens')
    print("vocab: ", vocab.get_vocab_size('labels'), vocab_dim)

    # 构建网络，此处网络为lstm-linear
    embedding_dim = 300
    hidden_dim = 128

    # 此处与demo_kaggle_jigsaw.py 中的随机embedding不同，glove目前只支持from_params，暂不支持构造函数实现
    glove_embeddings_file = '~/nlp/pretrainedEmbeddings/glove/glove.840B.300d.txt'
    token_embedding = Embedding.from_params(vocab=vocab,
                                            params=Params({'pretrained_file': glove_embeddings_file,
                                                           'embedding_dim': embedding_dim,
                                                           'trainable': False}))
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(embedding_dim, hidden_dim,
                                                  bidirectional=True,
                                                  batch_first=True))
    model = SSTClassifier(word_embeddings,
                          encoder,
                          out_dim=vocab.get_vocab_size("labels"),
                          vocab=vocab)

    # allennlp 目前好像不支持单机多卡，或者支持性能不好
    gpu_id = 0 if torch.cuda.is_available() else -1
    if gpu_id > -1:  model.cuda(gpu_id)

    # 构建迭代器，并为迭代器指定vocab
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    # -------- forward demo ---------
    generator = iter(iterator(train_dataset, shuffle=True))
    for _ in range(5):
        batch = next(generator)
        print('---\nbatch ', batch.keys(), batch['tokens'].keys(), batch['tokens']['tokens'].shape, batch['label'].shape) # [batch, sentence_len, token_len]
        batch = nn_util.move_to_device(batch, gpu_id)

        tokens = batch['tokens']
        mask = get_text_field_mask(tokens)
        embeddings = model.word_embeddings(tokens)
        print("embeddings: ", embeddings.shape)
        state = model.encoder(embeddings, mask)
        class_logits = model.linear(state)

        print("lstm state: ", state.shape, class_logits.shape)

        y = model(**batch)
        metric = model.get_metrics()
        print("model out: ", y, '\n', metric)

    # --------- train ------------
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # trainer = Trainer(model=model,
    #                   optimizer=optimizer,
    #                   iterator=iterator,
    #                   train_dataset=train_dataset,
    #                   validation_dataset=dev_dataset,
    #                   serialization_dir="./models/",
    #                   cuda_device=gpu_id,
    #                   patience=10,
    #                   num_epochs=20)
    # trainer.train()

    # ----------- predictor ---------------
    # TODO
    # predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    # logits = predictor.predict('This is the best movie ever!')['logits']
    # label_id = np.argmax(logits)
    # print(model.vocab.get_token_from_index(label_id, 'labels'))


if __name__ == "__main__":
    train_main()