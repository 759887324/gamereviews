
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from cnn_model import TCNNConfig, TextCNN
from data_loader import read_vocab, read_category, batch_iter, process_file
from config import *

model_name = 'textcnn'

save_path = save_path.format(model_name)
result_path = result_path.format(model_name)

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth=True
#生成字符表
vocab_size=5000
data=pd.read_csv('data/data1.csv','rb',engine='python')
data=list(data['comment,label_1,label_2,label_3'])

label=[]
contents=[]

for i in data:
    a=i.split(',')
    label.append(a[1])
    label.append(a[2])
    label.append(a[3])
    contents.append(a[0])
all_data = []
all_label = set(label)

for content in contents:
    all_data.extend(content)
counter = Counter(all_data)
count_pairs = counter.most_common(vocab_size - 1)
words, _ = list(zip(*count_pairs))
words = ['<PAD>'] + list(words)
all_label = list(all_label)
all_label=[x for x in all_label if x!='']
open(os.path.join(vocab_dir, 'vocab.txt'), mode='w').write('\n'.join(words) + '\n')
open(os.path.join(vocab_dir, 'label.txt'), mode='w').write('\n'.join(all_label) + '\n')



def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len




def train():
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    #配置Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    #载入训练集与验证集
    start_time = time.time()
    x_train, y_train, x_val, y_val,x_test,y_test = process_file('./data/train_data.csv', word_to_id, cat_to_id, config.seq_length)
    print(x_train.shape)
    print(len(y_train))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    #创建session
    session = tf.Session(config=tfconfig)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0

    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train, acc1 = session.run([model.loss, model.acc, model.acc1], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo
                #保存最好结果
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1


def test():
    print("Loading test data...")
    start_time = time.time()
    x_train, y_train, x_val,y_val,x_test, y_test = process_file('./data/train_data.csv', word_to_id, cat_to_id, config.seq_length)
    #x_test=x_train
    #y_test=y_train
    session = tf.Session(config=tfconfig)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  #读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 64
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1
    y_pred_cls = np.zeros(shape=[len(x_test), 6], dtype=np.int32)  # 保存预测结果
    y_pred_cls1 = np.zeros(shape=[len(x_test), 6], dtype=np.float32)  # 保存预测结果
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id], y_pred_cls1[start_id:end_id] = session.run([model.y_pred_cls, model.y_pred_cls1],
                                                                                feed_dict=feed_dict)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return y_pred_cls, y_pred_cls1, y_test

categories, cat_to_id = read_category(vocab_dir)
words, word_to_id = read_vocab(vocab_dir)
cat_to_id_1 = dict(zip(cat_to_id.values(), cat_to_id.keys()))

config = TCNNConfig()
config.vocab_size = len(words)
model = TextCNN(config)
#预测
def predict(sentence):
    session = tf.Session(config=tfconfig)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  #读取保存的模型
    sentence1=[]
    for i in sentence:
        sentence1.append(word_to_id[i])
    sentence1=sentence1+(200-len(sentence1))*[0]
    sentence1=[sentence1]
    feed_dict = {
            model.input_x:sentence1,
            model.keep_prob: 1.0
        }
    a,b,c = session.run([model.y_pred_cls,model.y_pred_cls1,model.logits], feed_dict=feed_dict)
    pred=list(1. / (1 + np.exp(-c)))
    pred=list(pred[0])
    print(pred)
    pred1=[pred.index(x) for x in pred if x>0.5]
    print(pred1)
    if len(pred1)>0:
        pred1=[cat_to_id_1[x] for x in pred1]
    else:
        pred1=np.argmax(pred)
        pred1=cat_to_id_1[pred1]
    return pred1



print('开始训练')
#train()
print('训练完成')

a, b, label = test()

c = np.zeros([a.shape[0], 6])
for i in range(a.shape[0]):
    if sum(a[i]) >= 1:
        c[i] = a[i]
    else:
        max1 = max(list(b[i]))
        index1 = list(b[i]).index(max1)
        c[i, index1] = 1

from sklearn.metrics import hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print('test-score')
print('accuracy',accuracy_score(label,c))
print('平均精确度：',average_precision_score(label, c))
print('汉明损失：',hamming_loss(label,c))
print('F1值：',f1_score(label,c,average='micro'))
print('召回率：',recall_score(label,c,average='micro'))
#print('混淆矩阵：',multilabel_confusion_matrix(label, c))
print(classification_report(label, c))
'''
#预测
data=pd.read_csv('')
for i in data:
    pred=predict(i)
    print(pred)


data = pd.read_csv('data/curse.csv')
data = list(data.iloc[:, 0])
pred1 = []
for i in data:
    pred = predict(i)
    print(i)
    print(pred)
    pred1.append(pred)

result = pd.DataFrame(data)
result['pred1'] = pred1
result.columns = ['输入', '预测值']
result.to_csv('result1.csv')
'''