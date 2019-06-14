"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: use_embeddings.py
@time: 2019/5/22 14:54
"""
from allennlp.common import Params
from allennlp.data import Vocabulary, Token, Instance
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.data.fields import TextField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


def batch_to_ids(batch, vocab):
    instances = []
    indexer = SingleIdTokenIndexer()
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'single_id': indexer})
        instance = Instance({"tokens": field})
        instances.append(instance)

    dataset = Batch(instances)
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['tokens']['single_id']

def use_glove():
    embedding_dim = 300
    project_dim = 200

    train_reader = StanfordSentimentTreeBankDatasetReader()
    dev_reader = StanfordSentimentTreeBankDatasetReader(use_subtrees=False)
    train_dataset = train_reader.read('~/nlp/dataset/sst/trees/train.txt')
    dev_dataset = dev_reader.read('~/nlp/dataset/sst/trees/dev.txt')

    print(f"total train samples: {len(train_dataset)}, dev samples: {len(dev_dataset)}")

    # 建立词汇表，从数据集中建立
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)
    vocab_dim = vocab.get_vocab_size('tokens')
    print("vocab: ", vocab.get_vocab_size('labels'), vocab_dim)

    glove_embeddings_file = '~/nlp/pretrainedEmbeddings/glove/glove.840B.300d.txt'

    # If you want to actually load a pretrained embedding file,
    # you currently need to do that by calling Embedding.from_params()
    # see https://github.com/allenai/allennlp/issues/2694
    token_embedding = Embedding.from_params(vocab=vocab,
                                            params=Params({'pretrained_file': glove_embeddings_file,
                                                           'embedding_dim': embedding_dim,
                                                           'trainable': False}))
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    print(word_embeddings.get_output_dim())

    # use batch_to_ids to convert sentences to character ids
    sentence_lists = [["I", 'have', 'a', "dog"], ["How", 'are' ,'you', ',', 'today', 'is', "Monday"]]

    sentence_ids = batch_to_ids(sentence_lists, vocab)
    embeddings = token_embedding(sentence_ids)

    for sentence in sentence_lists:
        for text in sentence:
            indice = vocab.get_token_index(text)
            print(f"text: {text}, indice: {indice}")

    # calculate distance based on elmo embedding
    import scipy
    tokens = [["dog", "ate", "an", "apple", "for", "breakfast"]]
    tokens2 = [["cat", "ate", "an", "carrot", "for", "breakfast"]]
    token_ids = batch_to_ids(tokens, vocab)
    token_ids2 = batch_to_ids(tokens2, vocab)
    vectors = token_embedding(token_ids)
    vectors2 = token_embedding(token_ids2)

    print('embedding shape ', vectors.shape)
    print('\nvector ', vectors[0][0], vectors2[0][0])
    distance = scipy.spatial.distance.cosine(vectors[0][0], vectors2[0][0])
    print(f"embedding distance: {distance}")




    # 构建迭代器，并为迭代器指定vocab
    # iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    # iterator.index_with(vocab)
    #
    # # -------- forward demo ---------
    # generator = iter(iterator(train_dataset, shuffle=True))
    # for _ in range(5):
    #     batch = next(generator)
    #     print('---\nbatch ', batch.keys(), batch['tokens'].keys(), batch['tokens']['tokens'].shape, batch['label'].shape) # [batch, sentence_len, token_len]





if __name__ == "__main__":
    use_glove()