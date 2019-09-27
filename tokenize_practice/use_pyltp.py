"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: tokenize.py
@time: 2019/9/5 13:04
"""


from pyltp  import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
#分句
with open("./emrData/liran/2246.txt", 'r') as f:
    text = f.read()
print(text)
sents = SentenceSplitter.split(text)
print('\n'.join(sents))

#中文分词
segmentor = Segmentor()  #初始化实例
segmentor.load("./models_v3.4.0/cws.model")  #加载模型
segmentor.load_with_lexicon("./models_v3.4.0/", "lexicon.txt")
words = segmentor.segment(text)  #分词


words = list(words)                                      #转换list
print(u"分词:", words)
segmentor.release()                                      #释放模型

#词性标注
pdir='./models_v3.4.0/pos.model'
pos = Postagger()                                        #初始化实例
pos.load(pdir)                                              #加载模型

postags = pos.postag(words)                        #词性标注
postags = list(postags)
print(u"词性:", postags)
pos.release()                                               #释放模型

data = {"words": words, "tags": postags}

for w, t in zip(words, postags):
    print(w, t)
#
# if __name__ == "__main__":
#     main()