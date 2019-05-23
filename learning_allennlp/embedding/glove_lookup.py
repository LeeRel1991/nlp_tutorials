"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: glove_lookup.py
@time: 2019/5/22 17:05
"""


from annoy import AnnoyIndex
import pickle

EMBEDDING_DIM = 300
GLOVE_FILE_PREFIX = '/home/lirui/nlp/document-qa/data/glove/glove.840B.300d{}'


def build_index():
    num_trees = 10

    idx = AnnoyIndex(EMBEDDING_DIM)

    index_to_word = {}
    with open(GLOVE_FILE_PREFIX.format('.txt')) as f:
        for i, line in enumerate(f):
            fields = line.rstrip().split(' ')
            vec = [float(x) for x in fields[1:]]
            idx.add_item(i, vec)
            index_to_word[i] = fields[0]
            # if i == 200000:
            #     break

    idx.build(num_trees)
    idx.save(GLOVE_FILE_PREFIX.format('.idx'))
    pickle.dump(index_to_word, open(GLOVE_FILE_PREFIX.format('.i2w'), mode='wb'))


def search(query, top_n=10):
    idx = AnnoyIndex(EMBEDDING_DIM)
    idx.load(GLOVE_FILE_PREFIX.format('.idx'))
    index_to_word = pickle.load(open(GLOVE_FILE_PREFIX.format('.i2w'), mode='rb'))
    word_to_index = {word: index for index, word in index_to_word.items()}
    query_id = word_to_index[query]
    print(query_id)
    word_ids = idx.get_nns_by_item(query_id, top_n)
    for word_id in word_ids:
        print(index_to_word[word_id])


if __name__ == '__main__':
    # build_index()
    search('dog')
    print()
    search('december')

    print('searcn for apple')
    search('apple')

    print('--search for carrot')
    search('carrot')
