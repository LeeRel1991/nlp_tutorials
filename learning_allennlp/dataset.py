"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: dataset.py
@time: 2019/5/4 16:02
"""
from pathlib import Path
from typing import *
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, MetadataField, ArrayField
import pandas as pd
import numpy as np

from overrides import overrides


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=True,
    seed=1,
    batch_size=8,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100,  # necessary to limit memory usage
    max_vocab_size=100000,
)


class JigsawDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = config.max_seq_len,
                 label_cols:List[str] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer

        # the token indexer is responsible for mapping tokens to integers
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len
        self.label_cols = label_cols

    @overrides
    def text_to_instance(self, tokens: List[Token], id: str = None,
                         labels: np.ndarray = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        id_field = MetadataField(id)
        fields["id"] = id_field

        if labels is None:
            labels = np.zeros(len(label_cols))
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)
        if config.testing: df = df.head(1000)
        for i, row in df.iterrows():
            tokens = [Token(x) for x in self.tokenizer(row["comment_text"])]
            yield self.text_to_instance(tokens, row["id"], row[self.label_cols].values)


from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer


def tokenizer(x: str):
    return [w.text for w in
            SpacyWordSplitter(language='en_core_web_sm',
                              pos_tags=False).split_words(x)[:config.max_seq_len]]


if __name__ == "__main__":

    label_cols = ["toxic", "severe_toxic", "obscene",
                  "threat", "insult", "identity_hate"]

    token_indexer = ELMoTokenCharactersIndexer()
    reader = JigsawDatasetReader(tokenizer=tokenizer,label_cols=label_cols)
    DATA_ROOT = Path("../data") / "jigsaw"
    train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["train.csv", "test_proced.csv"])
    val_ds = None
    total_num = len(train_ds)
    print(total_num)
    print(train_ds[0]['tokens']) # also use vars()

    vocab = Vocabulary.from_instances(train_ds, max_vocab_size=config.max_vocab_size)
    data_iterator = BucketIterator(batch_size=config.batch_size,
                              biggest_batch_first=True,
                              sorting_keys=[("tokens", "num_tokens")],
                             )
    # 为dataIteration指定词汇表，便于取索引id
    data_iterator.index_with(vocab)
    iterator = iter(data_iterator(train_ds))
    for _ in range(5):
        batch = next(iterator)
        print(batch['tokens']['tokens'].shape)


