"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: sst_classifier.py
@time: 2019/5/20 10:51
"""


# Model in AllenNLP represents a model that is trained.
from functools import reduce
from typing import Dict, Tuple, Optional

import torch
from overrides import overrides
from torch import nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("sst_classifier")
class SSTClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 embedding_dropout: float,
                 encoder: Seq2VecEncoder,
                 encoder_dropout: float,
                 out_dim: int,
                 vocab: Vocabulary,
                 positive_label: int = 4,
                 verbose=True) -> None:
        super().__init__(vocab)
        # 将word id 转为vector representations
        self._word_embeddings = word_embeddings
        self._embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self._encoder = encoder
        self._encoder_dropout = torch.nn.Dropout(encoder_dropout)

        # fc 层将上一层的维度转为输出的类别数
        self._linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                       out_features=out_dim)

        # 评价指标，分类准确率, F1 score
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_label)

        # 对于分类任务，交叉熵作为loss 函数
        # 而pytorch中的CrossEntropyLoss内部包含了一个softmax 和log likelihood loss，因此不必显示定义softmax层
        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.loss_function = torch.nn.BCEWithLogitsLoss()

        self._verbose = verbose

    def forward(self, tokens:Dict[str, torch.Tensor],
                label: torch.Tensor=None, id=None) -> Dict[str,torch.Tensor]:

        # 由于input时每个batch中的sentence 长度不尽相同，因此采用了b补0方式得到长度一致的sentence而转为Tensor进行forward，
        # 而实际forward中需要将补元素通过mask 方式处理
        mask = get_text_field_mask(tokens)

        embeddings = self._word_embeddings(tokens)

        if self._verbose:
            print("[sst classifier] embeddings out ", embeddings.shape)

        encoder_out = self._encoder(embeddings, mask)
        if self._verbose:
            print("[sst classifier] encoder out ", encoder_out.shape)
        logits = self._linear(encoder_out)

        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        acc = self.accuracy.get_metric(reset)
        return {"accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_measure": f1_measure}


@Model.register("multi_label_classifier")
class MultiLabelClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 embedding_dropout: float,
                 encoder: Seq2VecEncoder,
                 encoder_dropout: float,
                 out_dim: int,
                 vocab: Vocabulary,
                 verbose=False) -> None:
        super().__init__(vocab)
        # 将word id 转为vector representations
        self._word_embeddings = word_embeddings
        self._embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self._encoder = encoder
        self._encoder_dropout = torch.nn.Dropout(encoder_dropout)
        # fc 层将上一层的维度转为输出的类别数
        self._linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                       out_features=out_dim)

        # 评价指标，分类准确率, F1 score
        # self.accuracy = CategoricalAccuracy()
        # self.f1_measure = F1Measure(positive_label)

        # 对于分类任务，交叉熵作为loss 函数
        # 而pytorch中的CrossEntropyLoss内部包含了一个softmax 和log likelihood loss，因此不必显示定义softmax层
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self._verbose = verbose

    def forward(self, tokens:Dict[str, torch.Tensor],
                label: torch.Tensor=None, id=None) -> Dict[str,torch.Tensor]:

        # 由于input时每个batch中的sentence 长度不尽相同，因此采用了b补0方式得到长度一致的sentence而转为Tensor进行forward，
        # 而实际forward中需要将补元素通过mask 方式处理
        mask = get_text_field_mask(tokens)
        embeddings = self._word_embeddings(tokens)
        if self._verbose:
            print("[multi classifier] embeddings out ", embeddings.shape)

        dropped_embedded_text = self._embedding_dropout(embeddings)
        encoder_out = self._encoder(dropped_embedded_text, mask)
        dropped_encoder_out = self._encoder_dropout(encoder_out)
        if self._verbose:
            print("[multi classifier] encoder out ", encoder_out.shape)

        logits = self._linear(dropped_encoder_out)

        output = {"logits": logits}
        if label is not None:
            # self.accuracy(logits, label)
            # self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     precision, recall, f1_measure = self.f1_measure.get_metric(reset)
    #     acc = self.accuracy.get_metric(reset)
    #     return {"accuracy": acc,
    #             "precision": precision,
    #             "recall":recall,
    #             "f1_measure": f1_measure}




@Seq2VecEncoder.register("cnn_encoder")
class MyCnnEncoder(Seq2VecEncoder):
    """
    See Also https://github.com/dennybritz/cnn-text-classification-tf
    """
    def __init__(self,
                 embedding_dim: int,
                 num_filters: Tuple[int, ...] or int = 128,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),  # pylint: disable=bad-whitespace
                 conv_layer_activation: Activation = None,
                 output_dim: Optional[int] = None) -> None:
        super(MyCnnEncoder, self).__init__()
        self._embedding_dim = embedding_dim

        self._num_filters = num_filters
        if isinstance(self._num_filters, int):
            self._num_filters = [self._num_filters] * len(ngram_filter_sizes)

        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or Activation.by_name('relu')()
        self._output_dim = output_dim

        self._convolution_layers = []

        for i, (ngram_size, num) in enumerate(zip(self._ngram_filter_sizes, self._num_filters)):
            conv_layer = nn.Conv1d(in_channels=self._embedding_dim,
                                   out_channels=num,
                                   kernel_size=ngram_size)
            self._convolution_layers.append(conv_layer)
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim =  reduce(lambda x,y:x+y, self._num_filters)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(
                    self._activation(convolution_layer(tokens)).max(dim=2)[0]
            )

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result


class SkipCnnEncoder:
    pass


class DCnnEncoder:
    pass