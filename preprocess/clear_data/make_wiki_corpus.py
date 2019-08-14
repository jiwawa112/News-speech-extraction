#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import glob
import jieba
from opencc import OpenCC

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

file_path = 'F:/python/NLP_course/Project_01/wikiextractor/extracted/**/wiki_**'
files = glob.glob(file_path)
all_articles = []
for file in files:
    text = open(file,encoding='utf-8')
    lines = text.readlines()
    all_articles.append(lines)

print(all_articles[0])

# 过滤文本中的<>符号
pat = re.compile('<[^>]+>')
TEXT_1 = pat.sub('',str(all_articles))
TEXT_1 = TEXT_1.replace('n','')
print('过滤wiki格式完成！')
"""
def token():
    pat_1 = re.compile('<[^>]+>')
    # TEXT = pat.sub('',str(all_articles))
    pat_2 = re.compile('（）')
    pat_3 = re.compile('《》')
    pat_4 = re.compile('「')
    pat_5 = re.compile('」')
    for line in all_articles:
        line = pat_1.sub('', line)
        line = pat_2.sub('', line)
        line = pat_3.sub('', line)
        line = pat_4.sub('', line)
        line = pat_5.sub('', line)
"""

# 转换成简体中文
openCC = OpenCC('t2s')
TEXT_converted = openCC.convert(TEXT_1)
# print(TEXT_converted)

print('简体中文转化完毕!')
# 去噪
def token(string):
    return ' '.join(re.findall('[\w|\d]+',string))

TEXT = [token(str(a)) for a in TEXT_converted.split(' ') if a != 'n']

print('过滤文本完成！')

# 分词
def cut(string):
    return ' '.join(jieba.cut(string))

print('开始分词！')
with open('wiki_samples_KL.txt','w',encoding='utf-8') as f:
    for s in TEXT:
        f.write(cut(s)+'\n')

"""
# 追加训练模型
sentences = LineSentence('wiki_samples_AB.txt')
model = Word2Vec.load('word2vec.model')
model.train()
print(model.wv['说'])
print(model.wv.most_similar('说'))
print(model.wv.vocab)

model.save('wiki_and_chinese_news.model')
"""