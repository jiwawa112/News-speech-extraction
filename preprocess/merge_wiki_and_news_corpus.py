#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 获取目标文件路径
filedir = os.getcwd() + '/data'
print(filedir)

# 获取当前文件夹中文件名称
filenames = os.listdir(filedir)
print(filenames)

with open('wiki_and_news_data_lastest.txt','w',encoding='utf-8') as f:
    for filename in filenames:
        filepath = filedir + '/' + filename
        print('开始写入文件: {}'.format(filename))
        # 遍历单个文件，读取行数
        with open(filepath,'r',encoding='utf-8') as files:
            lines = files.readlines()
            for line in lines:
                f.writelines(line)
                f.write('\n')
