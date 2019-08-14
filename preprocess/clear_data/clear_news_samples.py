#!/usr/bin/env python3
# -*- coding: utf-8 -*-

data_path = 'F:/python/NLP_course/Project_01/make_wordvector_01/data/news_samples.txt'

def get_sentence(data_path):
    with open(data_path,'r',encoding='utf-8') as f:
        lines = f.read()
    return lines.replace('n','')
# print(get_sentence(data_path))

with open('news.txt','a',encoding='utf-8') as f:
    f.write(get_sentence(data_path))

with open('news.txt','r',encoding='utf-8') as f:
    text = f.read()

print(text)