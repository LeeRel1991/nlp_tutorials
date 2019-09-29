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
from utils import remove_special_chars, load_stopwords

def remove_stop_words(words_segmented, stopwords_list):

    outstr = ''
    for word in words_segmented:
        if word not in stopwords_list:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


def main():
    with open("./emrData/liran/2258.txt", 'r') as f:
        text = f.read()
    print(text)
    text = remove_special_chars(text)
    wordseg = list(jieba.cut(text, cut_all=False, HMM=False))
    print("默认词典: " + " ".join(wordseg))

    # -------------- 自定义词库达到更改默认分词的目的 -------------------------
    jieba.load_userdict("./models_v3.4.0/lexicon.txt")
    # jieba.set_dictionary('filename') # 重新设置词典
    wordseg = list(jieba.cut(text, cut_all=False, HMM=False))
    print("用户词典: " + " ".join(wordseg))

    # ------------------------- 去掉停用词 -------------------------
    stopwords = load_stopwords('./models_v3.4.0/stopWords.txt')  # 加载停用词列表
    wordseg_filtered = remove_stop_words(wordseg, stopwords)
    print("过滤停用词：", wordseg_filtered)

    # --------------- 词性标注 -------------------------
    wordseg_with_postag = pseg.cut(text)
    print("词性标注：", "/".join(["%s %s" % (w.word, w.flag) for w in wordseg_with_postag]))

    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    tagged = nltk.pos_tag(wordseg)  # 词性标注
    print("tagged", tagged)


    # ---------------- 在线修改词频以达到词语切分的目的 ------------------
    # test = "吴国忠臣伍子胥"
    # print("/".join(jieba.cut(test)))
    # print(jieba.get_FREQ("吴国忠"))
    # print(jieba.get_FREQ("臣"))
    # print(jieba.get_FREQ("吴国"))
    # print(jieba.get_FREQ("忠臣"))
    # jieba.add_word('忠臣',456) #更改忠臣词频为456
    # print("/".join(jieba.cut(test)))
    # text = "江州市长江大桥参加了长江大桥的通车仪式"
    # print("/".join(jieba.cut(text)))
    # jieba.load_userdict('./models_v3.4.0/lexicon.txt')
    # print("/".join(jieba.cut(text, HMM=False)))
    # print("/".join(jieba.cut(text, HMM=True)))


def count_word_freq():
    txt = open('./心脏病学.txt', "r")
    seg_txt = []
    f = open("./心脏病学_seg.txt", 'w')
    for line in txt:
        line = remove_special_chars(line)
        seg_list = jieba.lcut(line.strip('\n\r\t'))
        seg_txt.extend(seg_list)
        dst_lines = "#".join(seg_list)
        f.write(dst_lines+"\n")
    f.close()
    # 至此所有的中文词以list的形式存到了seg_txt中。


    # 下面进行词频排序，由高到底。
    word_dict = {}
    for item in seg_txt:
        if item not in word_dict:
            word_dict[item] = 1
        else:
            word_dict[item] += 1

    number = list(word_dict.items())
    number.sort(key=lambda x: x[1], reverse=True)
    print(len(number))
    i = 0

    with open("dict.txt", 'w') as fp:
        for w, count in number:
            fp.write("%s %d\n" % (w, count))


if __name__ == "__main__":
    main()
    # count_word_freq()
