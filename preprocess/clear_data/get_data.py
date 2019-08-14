#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymysql
import csv

# 打开数据库连接
db = pymysql.connect('cdb-q1mnsxjb.gz.tencentcdb.com','root','A1@2019@me','news_chinese',10102)
cursor = db.cursor()

# 使用 execute()方法执行SQL查询
cursor.execute("SELECT * FROM news_chinese.sqlResult_1558435")

# 读取数据
data = cursor.fetchall()

print(type(data))
print(len(data))
print(data[:10])
descre = cursor.description
print(cursor.description)
print(len(cursor.description))

cursor.close() # 关闭连接池
db.close()# 关闭数据库连接

with open('chinese_news.csv','w',newline='',encoding='utf-8') as f:
    # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
    writer = csv.writer(f,dialect='excel')
    # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
    header = []
    for i in descre:
        header.append(i[0])
    writer.writerow(header)
    for j in data:
        writer.writerow(j)

print('写入完成!')


