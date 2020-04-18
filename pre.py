import re # 正则表达式库
import collections # 词频统计库
import jieba
import pandas as pd
import numpy as np


pd.set_option('display.max_columns',10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth',10000)

csv = pd.read_csv('data3.csv',encoding='utf-8')
print(len(csv))

with open('stopwords.txt', encoding='utf-8', errors='ignore') as f:
    stops = f.read().split()

def fenci_1(x, stops):
    words = jieba.cut(x)
    word_nostop = []
    for c in words:
        if c not in stops:
            word_nostop.append(c)
    return ' '.join(word_nostop)
csv_1=csv['comment'].astype(str)
csv['comment']=csv_1.apply(lambda x:fenci_1(x,stops))

csv.to_csv(r'C:\Users\Administrator\Desktop\数据\data_pre1.csv',encoding='utf-8_sig')
