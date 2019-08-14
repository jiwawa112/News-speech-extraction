#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# 获取所有词频大于10的词
with open('similars_word_with_say_lastest.json','r') as f:
    words = json.load(f)
print(words)
print(len(words))

top10 = list()
for k,v in words:
    if v >= 10:
        top10.append(k)

print(top10)
print(len(top10))

