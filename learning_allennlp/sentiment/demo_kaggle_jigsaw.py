"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: demo_kaggle_jigsaw.py
@time: 2019/5/23 15:51
"""
from pathlib import Path

import allennlp
import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import move_to_device, get_text_field_mask
from allennlp.training import Trainer

from learning_allennlp.dataset import JigsawDatasetReader, custom_tokenizer


from learning_allennlp.sentiment.classifier import MultiLabelClassifier

label_cols = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]

def main():
    token_indexer = SingleIdTokenIndexer()
    reader = JigsawDatasetReader(tokenizer=custom_tokenizer(),
                                 token_indexers={"tokens": token_indexer},
                                 )

    # Kaggle的多标签“恶意评论分类挑战
    dataset_root = Path('../../data/jigsaw')
    train_dataset, dev_dataset = (reader.read(dataset_root/ fname) for fname in ["train.csv", "test_proced.csv"])

    print(f"total train samples: {len(train_dataset)}, dev samples: {len(dev_dataset)}")

    # 建立词汇表，从数据集中建立
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)
    vocab_dim = vocab.get_vocab_size('tokens')
    print("vocab: ", vocab.get_vocab_size('labels'), vocab_dim)

    # 构建网络，此处网络为lstm-linear
    embedding_dim = 300
    hidden_dim = 128
    token_embedding = Embedding(num_embeddings=vocab_dim,
                                embedding_dim=embedding_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(embedding_dim, hidden_dim,
                                                  bidirectional=True,
                                                  batch_first=True))
    model = MultiLabelClassifier(word_embeddings,
                                 0.5,
                                 encoder,
                                 0.2,
                                 len(label_cols), vocab)

    # allennlp 目前好像不支持单机多卡，或者支持性能不好
    gpu_id = 0 if torch.cuda.is_available() else -1
    if gpu_id > -1:  model.cuda(gpu_id)


    # 构建迭代器，并为迭代器指定vocab
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    # --------------------- forward demo ----------------------
    # generator = iter(iterator(train_dataset, shuffle=True))
    # for _ in range(5):
    #     batch = next(generator)
    #     print('---\nbatch ', batch.keys(), batch['tokens'].keys(), batch['tokens']['tokens'].shape, batch['label'].shape) # [batch, sentence_len, token_len]
    #     batch = move_to_device(batch, gpu_id)
    #     tokens = batch['tokens']
    #
    #     # option1. forward one step by one
    #     mask = get_text_field_mask(tokens)
    #     embeddings = model.word_embeddings(tokens)
    #     print("embeddings: ", embeddings.shape)
    #     state = model.encoder(embeddings, mask)
    #     class_logits = model.linear(state)
    #
    #     print("lstm state: ", state.shape, class_logits.shape)
    #
    #     # option2. do forward on the model
    #     y = model(**batch)
    #     metric = model.get_metrics()
    #     print("model out: ", y, '\n', metric)

    # --------------------- train ---------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
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


if __name__ == "__main__":
    main()