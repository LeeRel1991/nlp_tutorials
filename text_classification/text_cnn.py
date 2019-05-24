"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: text_cnn.py
@time: 2019/5/24 17:46
"""
from typing import Dict

import torch
import torch.nn as nn
from allennlp.common.checks import ConfigurationError
from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules import Embedding

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


class CnnClassifier(Model):
    def __init__(self, vocab, cnn_options):
        super(CnnClassifier, self).__init__(vocab)

        filters = cnn_options['filters']
        char_embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim,
                out_channels=num,
                kernel_size=width,
                bias=True
            )

            convolutions.append(conv)
            self.add_module('conv_{}'.format(i), conv)

        self._convolutions = convolutions
        self._pool = torch.nn.MaxPool2d()
        if cnn_options['activation'] == 'tanh':
            self._activation = torch.tanh
        elif cnn_options['activation'] == 'relu':
            self._activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

    def forward(self, tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        pass


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


def main():
    token_indexer = SingleIdTokenIndexer()

    token_embedding = Embedding(num_embeddings=100,
                                embedding_dim=50)

    # text_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    sequence_ids = torch.Tensor([[4, 2, 67, 54, 30, 9, 0, 0], [87, 43, 12, 25, 81, 24, 52, 70]])
    print("seq bath: ", sequence_ids.size())

    sequence_embedding = token_embedding(sequence_ids)
    print("embessing ", sequence_embedding.size())


if __name__ == "__main__":
    main()
