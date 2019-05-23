"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: sst_classifier.py
@time: 2019/5/20 10:51
"""


# Model in AllenNLP represents a model that is trained.
from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("sst_classifier")
class SSTClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 out_dim: int,
                 vocab: Vocabulary,
                 positive_label: int = 4) -> None:
        super().__init__(vocab)
        # 将word id 转为vector representations
        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # fc 层将上一层的维度转为输出的类别数
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=out_dim)

        # 评价指标，分类准确率, F1 score
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_label)

        # 对于分类任务，交叉熵作为loss 函数
        # 而pytorch中的CrossEntropyLoss内部包含了一个softmax 和log likelihood loss，因此不必显示定义softmax层
        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.loss_function = torch.nn.BCEWithLogitsLoss()

    def forward(self, tokens:Dict[str, torch.Tensor],
                label: torch.Tensor=None, id=None) -> Dict[str,torch.Tensor]:

        # 由于input时每个batch中的sentence 长度不尽相同，因此采用了b补0方式得到长度一致的sentence而转为Tensor进行forward，
        # 而实际forward中需要将补元素通过mask 方式处理
        mask = get_text_field_mask(tokens)

        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

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
                 encoder: Seq2VecEncoder,
                 out_dim: int,
                 vocab: Vocabulary,
                 positive_label: int = 4) -> None:
        super().__init__(vocab)
        # 将word id 转为vector representations
        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # fc 层将上一层的维度转为输出的类别数
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=out_dim)

        # 评价指标，分类准确率, F1 score
        # self.accuracy = CategoricalAccuracy()
        # self.f1_measure = F1Measure(positive_label)

        # 对于分类任务，交叉熵作为loss 函数
        # 而pytorch中的CrossEntropyLoss内部包含了一个softmax 和log likelihood loss，因此不必显示定义softmax层
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def forward(self, tokens:Dict[str, torch.Tensor],
                label: torch.Tensor=None, id=None) -> Dict[str,torch.Tensor]:

        # 由于input时每个batch中的sentence 长度不尽相同，因此采用了b补0方式得到长度一致的sentence而转为Tensor进行forward，
        # 而实际forward中需要将补元素通过mask 方式处理
        mask = get_text_field_mask(tokens)

        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

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
