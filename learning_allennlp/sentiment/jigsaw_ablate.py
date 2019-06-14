"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: jigsaw_ablate.py
@time: 2019/5/31 14:18
"""
from pathlib import Path

from learning_allennlp.dataset import JigsawDatasetReader, custom_tokenizer

import logging
from argparse import ArgumentParser

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader as sstDataReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.models import BiattentiveClassificationNetwork
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training import Trainer

from learning_allennlp.sentiment.classifier import SSTClassifier, MultiLabelClassifier

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)


def get_args():
    parser = ArgumentParser(description='train a ablative model on kaggle jigsaw')
    parser.add_argument('--embedding', type=str, default='glove',
                        help='embedding type, [random, word2vec, glove, cove, elmo]')
    parser.add_argument('--embedding_dim', type=int, default=100)

    parser.add_argument('--encoder', type=str, default='lstm', help='encoder network, [lstm, cnn]')

    parser.add_argument('--network', type=str, default=None, help='network , [None, bcn]')

    parser.add_argument('--train_data', type=str, default='~/nlp/dataset/sst/trees/train.txt', help='path for trainset')
    parser.add_argument('--dev_data', type=str, default='~/nlp/dataset/sst/trees/dev.txt', help='path for devset')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=3)

    return parser.parse_args()


def main():
    args = get_args()

    # TODO 增加char n-gram embeddings
    if args.embedding == 'elmo':
        token_indexer = ELMoTokenCharactersIndexer()
    else:
        token_indexer = SingleIdTokenIndexer()
    # Kaggle的多标签“恶意评论分类挑战
    reader = JigsawDatasetReader(tokenizer=None,
                                 token_indexers={"tokens": token_indexer},
                                 max_seq_len=200)

    dataset_root = Path('../../data/jigsaw')
    train_dataset, dev_dataset = (reader.read(dataset_root / fname) for fname in ["train.csv", "test_proced.csv"])

    print(f"total train samples: {len(train_dataset)}, dev samples: {len(dev_dataset)}")

    # 建立词汇表，从数据集中建立
    # if args.embedding == 'elmo':
    #     vocab = Vocabulary()
    # else:
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

    vocab_dim = vocab.get_vocab_size('tokens')
    print("vocab: ", vocab.get_vocab_size('labels'), vocab_dim)

    # 建立token embedding
    token_embedding = None
    print(f"embedding dim: {args.embedding_dim}")
    if args.embedding == 'random':
        token_embedding = Embedding(num_embeddings=vocab_dim,
                                    embedding_dim=args.embedding_dim)
    elif args.embedding == 'glove':
        glove_embeddings_file = '~/nlp/pretrainedEmbeddings/glove/glove.6B.100d.txt'
        token_embedding = Embedding.from_params(vocab=vocab,
                                                params=Params({'pretrained_file': glove_embeddings_file,
                                                               'embedding_dim': args.embedding_dim,
                                                               'trainable': False}))
    elif args.embedding == 'elmo':
        # pretrained elmo LM model, transformed from bilm-tf with dump_weights in bin/training.py
        options_file = '~/nlp/pretrainedEmbeddings/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        weight_file = '~/nlp/pretrainedEmbeddings/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

        token_embedding = ElmoTokenEmbedder(options_file, weight_file,
                                            requires_grad=True,
                                            do_layer_norm=False
                                            )

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    if args.embedding == 'elmo':
        args.embedding_dim = word_embeddings.get_output_dim()

    # 建立encoder seq2vec
    if args.encoder == 'lstm':
        hidden_dim = 256
        encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(args.embedding_dim, hidden_dim,
                                                      bidirectional=True,
                                                      batch_first=True))
    elif args.encoder == 'cnn':
        encoder = CnnEncoder(embedding_dim=args.embedding_dim,
                             num_filters=128,
                             ngram_filter_sizes=(2, 3, 4, 5, 6, 7),
                             )
    else:
        encoder = None

    # 建立 主分类网络
    if args.network is None:
        model = MultiLabelClassifier(word_embeddings,
                                     0.5,
                                     encoder,
                                     0.2,
                                     vocab=vocab,
                                     out_dim=6,
                                     )
    elif args.network == 'bcn':
        # TODO 转换成code line 形式 实例化分类器网络
        bcn_params = {"text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "pretrained_file": "/home/lirui/nlp/document-qa/data/glove/glove.840B.300d.txt",
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": False
                }
            }
        },
            "embedding_dropout": 0.5,
            "pre_encode_feedforward": {
                "input_dim": 300,
                "num_layers": 1,
                "hidden_dims": [300],
                "activations": ["relu"],
                "dropout": [0.25]
            },
            "encoder": {
                "type": "lstm",
                "input_size": 300,
                "hidden_size": 300,
                "num_layers": 1,
                "bidirectional": True
            },
            "integrator": {
                "type": "lstm",
                "input_size": 1800,
                "hidden_size": 300,
                "num_layers": 1,
                "bidirectional": True
            },
            "integrator_dropout": 0.1,
            # "elmo": {
            #     "options_file": "/home/lirui/nlp/learning_allenNLP/learning_allennlp/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            #     "weight_file": "/home/lirui/nlp/learning_allenNLP/learning_allennlp/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            #     "do_layer_norm": False,
            #     "dropout": 0.0,
            #     "num_output_representations": 1
            # },
            # "use_input_elmo": False,
            # "use_integrator_output_elmo": False,
            "output_layer": {
                "input_dim": 2400,
                "num_layers": 3,
                "output_dims": [1200, 600, 5],
                "pool_sizes": 4,
                "dropout": [0.2, 0.3, 0.0]
            }
        }
        model = BiattentiveClassificationNetwork.from_params(vocab,
                                                             params=Params(bcn_params))

    # 训练参数
    gpu_id = args.gpu_id if torch.cuda.is_available() else -1
    if gpu_id > -1:  model.cuda(gpu_id)

    # 构建迭代器，并为迭代器指定vocab
    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

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
                      num_epochs=args.n_epochs)
    trainer.train()


if __name__ == "__main__":
    main()
