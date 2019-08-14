#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

# 训练词向量模型
sentences = LineSentence('wiki_and_news_data_lastest.txt')
model = Word2Vec(sentences,size=200,min_count=1,workers=4)
print(model.wv['说'])
# print(model.wv.most_similar('说'))
# print(model.wv.vocab)

model.save('wiki_and_news_word2vec_200_lastest.model')
