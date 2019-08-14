#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob

file_path = 'F:/python/NLP_course/Project_01/wikiextractor/extracted/**wiki_**'
files = glob.glob(file_path)

all_artices = []

for file in files:
    text = open(file,encoding='utf-8')
    lines = text.readlines()
    all_artices.append(lines)

print(all_artices[0])

# 过滤文本中的<>
pat = re.compile('<[^>]+>')
TEXT = pat.sub('',str(all_articles))

