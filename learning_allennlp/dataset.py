"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: dataset.py
@time: 2019/5/4 16:02
"""
from pathlib import Path
from typing import *
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Vocabulary, Tokenizer
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField
import pandas as pd
import numpy as np
from allennlp.data.tokenizers import WordTokenizer

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
    """
    Kaggle的多标签“恶意评论分类挑战”
    目标标签共5类：普通恶评、严重恶评、污言秽语、威胁、侮辱和身份仇视

    csv 文件格式：
    每行一个样本，不同列之间用逗号隔开，第一行为标题行，类别对应的0表示无，1表示有，一个样本可能对应多个标签
    "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
    "000103f0d9cfb60f","D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)",0,0,0,0,0,0

    """

    def __init__(self, tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = None,
                 ) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        # the token indexer is responsible for mapping tokens to integers
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.max_seq_len = max_seq_len
        self.label_cols: List[str] = None

    @overrides
    def text_to_instance(self, comment_text: str, id: str = None,
                         labels: np.ndarray = None) -> Instance:
        tokens = self._tokenizer.tokenize(comment_text)
        if self.max_seq_len is not None:
            tokens = tokens[:self.max_seq_len]

        sentence_field = TextField(tokens, self._token_indexers)
        fields = {"tokens": sentence_field}

        id_field = MetadataField(id)
        fields["id"] = id_field

        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)

        # csv 文件的第一行为标题，id text cls1 ...
        self.label_cols = df.keys()[2:]
        # print("title ", self.label_cols)

        for i, row in df.iterrows():
            yield self.text_to_instance(row["comment_text"], row["id"], row[self.label_cols].values)


class MovieReviewDatasetReader(DatasetReader):
    """
    sentence polarity dataset v1.0 from http://www.cs.cornell.edu/people/pabo/movie-review-data/
    Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews
    Notes:从官网上下载的mr样本包含两个文件"rt-polarity.pos", "rt-polarity.neg"，分别表示正样本和负样本，需经过mrdataset_preprocess.py
    对其进行预处理，分割为train 和test 两个子集并保存（train.pkl和test.pkl），才能使用该类进行读取加载
    """
    def __init__(self, tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = None,
                 ) -> None:
        super(MovieReviewDatasetReader, self).__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        # the token indexer is responsible for mapping tokens to integers
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._max_seq_len = max_seq_len

    def text_to_instance(self, comment_text: str,
                         sentiment: int = None) -> Instance:

        tokens = self._tokenizer.tokenize(comment_text)
        if self._max_seq_len is not None:
            tokens = tokens[:self._max_seq_len]

        sentence_field = TextField(tokens, self._token_indexers)
        fields = {"tokens": sentence_field}

        label_field = LabelField(str(sentiment))
        fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Args:
            file_path: train.pkl or test.pkl generated by mrdataset_preprocess.py. 文件存储了train 和test的样本列表（a list of dict），、
            列表中每个元素是一个dict，形如：
            {"y": 0,    "text": orig_rev,  "num_words": len(orig_rev.split()),    "split": np.random.randint(0, cv)}
            其中key 'split' 表示采用10-cv 法进行train test 分割时所对应的序号

        Returns:

        """
        # TODO: padding for cnn
        import pickle, os
        file_path = os.path.expanduser(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        for rev in data:
            yield self.text_to_instance(rev['text'], rev['y'])


from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer


def custom_tokenizer():
    return WordTokenizer(word_splitter=SpacyWordSplitter(language='en_core_web_sm', pos_tags=False))


if __name__ == "__main__":
    token_indexer = ELMoTokenCharactersIndexer()
    # reader = JigsawDatasetReader(max_seq_len=100)
    # DATA_ROOT = Path("../data") / "jigsaw"
    # train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["train.csv", "test_proced.csv"])

    reader = MovieReviewDatasetReader(max_seq_len=100)
    DATA_ROOT = Path("/home/lirui/nlp/dataset/MRv1.0") / "rt-polaritydata"
    train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["train.pkl", "test.pkl"])

    total_num = len(train_ds)
    print(total_num)
    print(train_ds[0]['tokens'])  # also use vars()

    vocab = Vocabulary.from_instances(train_ds)
    vocab_dim = vocab.get_vocab_size('tokens')
    out_dim = vocab.get_vocab_size('labels')
    print("vocab: ", vocab.get_vocab_size('labels'), vocab_dim)

    data_iterator = BucketIterator(batch_size=64,
                                   biggest_batch_first=True,
                                   sorting_keys=[("tokens", "num_tokens")],
                                   )
    # # 为dataIteration指定词汇表，便于取索引id
    data_iterator.index_with(vocab)
    iterator = iter(data_iterator(train_ds))
    for _ in range(int(len(train_ds)/64)):
        batch = next(iterator)
        print("iter ", batch['tokens']['tokens'].shape)
