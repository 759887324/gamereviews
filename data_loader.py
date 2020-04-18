
import numpy as np
import tensorflow.keras as kr
import os
import pandas as pd
#读取文件
def read_file():
    data=pd.read_csv('data/data3.csv','rb',engine='python')
    data = data.sample(frac=1, random_state=2020)
    data=list(data['comment,label_1,label_2,label_3'])
    
    label=[]
    contents=[]
    for i in data:
        a=i.split(',')
        
        if a[2]=='':
            label.append(a[1:2])
        elif a[3]=='':
            label.append(a[1:3])
        else:
            label.append(a[1:])
        contents.append(a[0])
        
    return contents, label

#读取字符表，并转换为id表示
def read_vocab(vocab_dir):
    
    with open(os.path.join(vocab_dir, 'vocab.txt')) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

#读取类别表，并转换为id表示
def read_category(vocab_dir):

    with open(os.path.join(vocab_dir, 'label.txt')) as fp:
        categories = [_.strip() for _ in fp.readlines()]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

#将id转换为文字表示
def to_words(content, words):
    
    return ''.join(words[x] for x in content)

#分割数据集
def process_file(filename, word_to_id, cat_to_id, max_length=200):
    
    contents, labels = read_file()

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append([cat_to_id[x] for x in labels[i]])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    class_matrix = np.eye(len(cat_to_id))
    y_pad = np.array(list(map(lambda x: np.sum(class_matrix[x], axis=0), label_id)))
    x_pad_train=x_pad[0:7000]
    x_pad_val = x_pad[7000:9000]
    x_pad_test=x_pad[9000:]

    y_pad_train=y_pad[0:7000]
    y_pad_val=y_pad[7000:9000]
    y_pad_test=y_pad[9000:]
    return x_pad_train, y_pad_train,x_pad_val,y_pad_val,x_pad_test,y_pad_test


#批次处理
def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
