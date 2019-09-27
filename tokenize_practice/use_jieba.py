"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: use_jieba.py
@time: 2019/9/5 14:14
"""
import jieba
import nltk
import re
import jieba.posseg as pseg


def remove_special_chars(text):
    # 利用正则表达式去掉一些一些标点符号之类的符号。
    text = re.sub(r'\s+', ' ', text)  # trans 多空格 to空格
    text = re.sub(r'\n+', ' ', text)  # trans 换行 to空格
    text = re.sub(r'\t+', ' ', text)  # trans Tab to空格
    text = re.sub(r' +', '', text)  # trans 删除空格
    # text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+",
    #               "", text)
    print(text)
    return text


# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def main():
    with open("./emrData/liran/2258.txt", 'r') as f:
        text = f.read()
    print(text)
    text = remove_special_chars(text)
    jieba.load_userdict("./models_v3.4.0/lexicon.txt")

    stopwords = stopwordslist('./models_v3.4.0/stopWords.txt')  # 这里加载停用词的路径
    wordseg = list(jieba.cut(text, cut_all=False))
    wordseg_with_postag = pseg.cut(text)
    print("精确模式: " + " ".join(wordseg))
    # for w in wordseg_with_postag:
    #     print(w.word, w.flag)
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    tagged = nltk.pos_tag(wordseg)  # 词性标注
    print("tagged", tagged)
    entities = nltk.chunk.ne_chunk(tagged)  # 命名实体识别
    a1 = str(entities)
    print(a1)

    # print(len(list(wordseg))
    outstr = ''
    # for word in wordseg:
    #     if word not in stopwords:
    #         if word != '\t':
    #             outstr += word
    #             outstr += " "
    # print("去掉停用词 ",outstr)


#

if __name__ == "__main__":
    main()
