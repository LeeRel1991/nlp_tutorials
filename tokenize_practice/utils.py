import re


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
def load_stopwords(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
