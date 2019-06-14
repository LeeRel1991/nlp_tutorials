"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: text_cnn.py
@time: 2019/5/24 17:46
"""
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules import Embedding, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import CnnEncoder

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.nn.util import move_to_device, get_text_field_mask
from allennlp.training import Trainer
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from learning_allennlp.dataset import JigsawDatasetReader, custom_tokenizer, MovieReviewDatasetReader
from learning_allennlp.sentiment.sst_classifier import SSTClassifier, MultiLabelClassifier
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader as sstDataReader
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

@Model.register("cnn_classifier")
class CnnClassifier(Model):
    def __init__(self, vocab,
                 text_field_embedder: TextFieldEmbedder,
                 cnn_options):
        super(CnnClassifier, self).__init__(vocab)

        self._vocab = vocab
        self._text_field_embedder = text_field_embedder
        filters = cnn_options['filters']
        embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num,
                kernel_size=width,
                bias=True
            )

            convolutions.append(conv)
            self.add_module('conv_{}'.format(i), conv)

        self._convolutions = convolutions
        # self._pool = torch.nn.MaxPool2d()
        if cnn_options['activation'] == 'tanh':
            self._activation = torch.tanh
        elif cnn_options['activation'] == 'relu':
            self._activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # fc 层将上一层的维度转为输出的类别数
        self.linear = torch.nn.Linear(in_features=2048,
                                      out_features=5)

        # 评价指标，分类准确率, F1 score
        self.accuracy = CategoricalAccuracy()
        # self.f1_measure = F1Measure(positive_label)

        # 对于分类任务，交叉熵作为loss 函数
        # 而pytorch中的CrossEntropyLoss内部包含了一个softmax 和log likelihood loss，因此不必显示定义softmax层
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        text_embed = self._text_field_embedder(tokens)
        text_embed = torch.transpose(text_embed, 1, 2)

        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'conv_{}'.format(i))
            convolved = conv(text_embed)
            print(f"\nconv{i}: after conv {convolved.shape}")
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            print(f"conv{i}: after maxpool {convolved.shape}")

            convolved = self._activation(convolved)
            convs.append(convolved)
        token_embedding = torch.cat(convs, dim=-1)

        logits = self.linear(token_embedding)

        output = {"embeddings": token_embedding,
                  "logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        acc = self.accuracy.get_metric(reset)
        return {"accuracy": acc,
                # "precision": precision,
                # "recall": recall,
                # "f1_measure": f1_measure
                }


def batch_to_ids(batch):
    instances = []
    indexer = SingleIdTokenIndexer()
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'single_id': indexer})
        instance = Instance({"tokens": field})
        instances.append(instance)

    vocab = Vocabulary.from_instances(instances)
    dataset = Batch(instances)
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['tokens']['single_id']


label_cols = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]


def main():
    token_indexer = SingleIdTokenIndexer()
    # train_reader = sstDataReader(token_indexers={'tokens': token_indexer}, use_subtrees=False)
    # dev_reader = sstDataReader(token_indexers={'tokens': token_indexer}, use_subtrees=False)
    # dataset_root = Path("~/nlp/dataset/sst/trees")
    # train_dataset, dev_dataset = (reader.read(dataset_root / fname) for reader, fname in zip([train_reader, dev_reader], ["train.txt", "dev.txt"]))

    reader = MovieReviewDatasetReader(max_seq_len=200)
    dataset_root = Path("/home/lirui/nlp/dataset/MRv1.0/rt-polaritydata")
    train_dataset, dev_dataset = (reader.read(dataset_root / fname) for fname in ["train.pkl", "test.pkl"])

    # Kaggle的多标签“恶意评论分类挑战
    # reader = JigsawDatasetReader(tokenizer=None,
    #                              token_indexers={"tokens": token_indexer},
    #                              max_seq_len=200)
    # dataset_root = Path('../data/jigsaw')
    # train_dataset, dev_dataset = (reader.read(dataset_root / fname) for fname in ["train.csv", "test_proced.csv"])

    print(f"total train samples: {len(train_dataset)}, dev samples: {len(dev_dataset)}")

    # 建立词汇表，从数据集中建立
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

    vocab_dim = vocab.get_vocab_size('tokens')
    print("vocab: ", vocab.get_vocab_size('labels'), vocab_dim)

    embedding_dim = 300
    # token_embedding = Embedding(num_embeddings=vocab_dim,  embedding_dim=embedding_dim)

    # 此处与随机embedding不同，glove目前只支持from_params，暂不支持构造函数实现
    glove_embeddings_file = '~/nlp/pretrainedEmbeddings/glove/glove.840B.300d.txt'
    token_embedding = Embedding.from_params(vocab=vocab,
                                            params=Params({'pretrained_file': glove_embeddings_file,
                                                           'embedding_dim': embedding_dim,
                                                           'trainable': False}))
    text_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    encoder = CnnEncoder(embedding_dim=embedding_dim,
                         num_filters=200,
                         ngram_filter_sizes=(3, 4, 5),
                         )

    # model = MultiLabelClassifier(text_embedder,
    #                              0,
    #                              encoder,
    #                              0.2,
    #                              vocab=vocab,
    #                              out_dim=6,
    #                              )

    model = SSTClassifier(text_embedder, 0.2, encoder, 0.2, vocab.get_vocab_size('labels'), vocab, verbose=False)
    # sequence_ids = torch.Tensor([[4, 2, 67, 54, 30, 9, 0, 0], [87, 43, 12, 25, 81, 24, 52, 70]]).long()
    # sequence_ids.cuda()
    # print("seq bath: ", sequence_ids.size())
    # sequence_embedding = token_embedding(sequence_ids)
    # print("embedding ", sequence_embedding.size())
    # y = model({"tokens": sequence_ids})
    # print("model out ", y.keys(), y['logits'].shape)

    # 训练参数
    gpu_id = 1 if torch.cuda.is_available() else -1
    if gpu_id > -1:
        model.cuda(gpu_id)

    # 构建迭代器，并为迭代器指定vocab
    iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)


    # -------- forward demo ---------
    # generator = iter(iterator(train_dataset, shuffle=False))
    # for _ in range(10):
    #     batch = next(generator)
    #     print('---\nbatch ', batch.keys(), batch['tokens'].keys(), batch['tokens']['tokens'].shape, batch['label'].shape) # [batch, sentence_len, token_len]
    #     batch = move_to_device(batch, gpu_id)

    #     tokens = batch['tokens']
    #     mask = get_text_field_mask(tokens)
    #     embeddings = model.word_embeddings(tokens)
    #     print("embeddings: ", embeddings.shape)
    #     state = model.encoder(embeddings, mask)
    #     class_logits = model.linear(state)
    #
    #     print("lstm state: ", state.shape, class_logits.shape)

        # y = model(**batch)
        # metric = model.get_metrics()
        # print("model out: ", y, '\n', metric)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      grad_norm=5.0,
                      # validation_metric='+accuracy',
                      cuda_device=gpu_id,
                      patience=5,
                      num_epochs=20)
    trainer.train()


if __name__ == "__main__":
    main()
