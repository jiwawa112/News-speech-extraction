#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

import re
import jieba
import pandas as pd

# sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
# model = Word2Vec(sentences,min_count=1)

# print(model.wv.vocab)
# print(model.most_similar('cat'))
# print(model.wv['dog'])

content = pd.read_csv('chinese_news.csv',encoding='utf-8')
samples = content['content']
# print(samples)

def tokens(string):
    return ' '.join(re.findall('[\w|\d]+', string))

valiad_samples = [tokens(str(a)) for a in samples if a!= 'n']

def cut(string):return ' '.join(jieba.cut(string))

with open('news_samples.txt','w',encoding='utf-8') as f:
    for s in valiad_samples:
        f.write(cut(s)+'\n')


