"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: sst_lstm_elmo.py
@time: 2019/5/20 11:57
"""
import logging
from pathlib import Path

import torch
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.modules import Elmo
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training import Trainer
from allennlp.nn import util as nn_util
from learning_allennlp.dataset import JigsawDatasetReader, tokenizer, Config
from learning_allennlp.sentiment.sst_classifier import SSTClassifier, MultiLabelClassifier

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)


def train_main():
    config = Config(
        testing=True,
        seed=1,
        batch_size=64,
        lr=3e-4,
        epochs=2,
        hidden_sz=64,
        max_seq_len=100,  # necessary to limit memory usage
        max_vocab_size=100000,
    )
    token_indexer = ELMoTokenCharactersIndexer()
    # 目标标签，普通恶评、严重恶评、污言秽语、威胁、侮辱和身份仇视
    # label_cols = ["toxic", "severe_toxic", "obscene",
    #               "threat", "insult", "identity_hate"]
    # reader = JigsawDatasetReader(tokenizer=tokenizer,
    #                              token_indexers={"tokens": token_indexer},
    #                              label_cols=label_cols)

    # Kaggle的多标签“恶意评论分类挑战
    # dataset_root = Path('/home/lirui/nlp/learning_allenNLP/data/jigsaw')
    # train_dataset, dev_dataset = (reader.read(dataset_root/ fname) for fname in ["train.csv", "test_proced.csv"])

    # stanford  情感分类-sst5 数据集
    reader = StanfordSentimentTreeBankDatasetReader(token_indexers={'tokens': token_indexer})
    train_dataset = reader.read('~/nlp/dataset/sst/trees/train.txt')
    dev_dataset = reader.read('~/nlp/dataset/sst/trees/test.txt')

    print(f"total train samples: {len(train_dataset)}, dev samples: {len(dev_dataset)}")

    # 建立词汇表，
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

    # pretrained elmo LM model, transformed from bilm-tf with dump_weights in bin/training.py
    options_file = '../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = '../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    token_embedding = ElmoTokenEmbedder(options_file, weight_file,
                                        requires_grad=True,
                                        # do_layer_norm=True
                                        )

    # Pass in the ElmoTokenEmbedder instance instead
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    elmo_embedding_dim = word_embeddings.get_output_dim()
    hidden_dim = 256
    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(elmo_embedding_dim, hidden_dim, bidirectional=True,batch_first=True))

    model = SSTClassifier(word_embeddings,
                          encoder,
                          out_dim=vocab.get_vocab_size("labels"),
                          vocab=vocab)

    gpu_id = 0 if torch.cuda.is_available() else -1
    if gpu_id > -1:  model.cuda(gpu_id)

    # 构建迭代器，并为迭代器指定vocab
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    # -------- forward demo ---------
    # generator = iter(iterator(train_dataset, shuffle=True))
    # for _ in range(5):
    #     batch = next(generator) # [batch, sentence_len, token_len]
    #     print('---\nbatch ', batch.keys(), batch['tokens'].keys(), batch['tokens']['tokens'].shape, batch['label'].shape)
    #     batch = nn_util.move_to_device(batch, 0 if use_gpu else -1)
    #
    #     tokens = batch['tokens']
    #     mask = get_text_field_mask(tokens)
    #     embeddings = model.word_embeddings(tokens)
    #     print("embeddings: ", embeddings.shape)
    #     state = model.encoder(embeddings, mask)
    #     class_logits = model.linear(state)
    #
    #     print("lstm state: ", state.shape, class_logits.shape)
    #
    #     y = model(**batch)
    #     print("model out: ", y)
    #
    # print("\nparams ")
    # for n, p in model.named_parameters():
    #     print(n, p.size())

    # --------- train ------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      # serialization_dir="./models/",
                      cuda_device=gpu_id,
                      patience=10,
                      num_epochs=20)

    trainer.train()

    # tokens = ['This', 'is', 'the', 'best', 'movie', 'ever', '!']
    # predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    # logits = predictor.predict(tokens)['logits']
    # label_id = np.argmax(logits)
    # print(model.vocab.get_token_from_index(label_id, 'labels'))


if __name__ == "__main__":
    train_main()