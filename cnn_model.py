
import tensorflow as tf
from config import num_classes


class TCNNConfig(object):
    embedding_dim = 64  #字向量维度
    seq_length = 200  #序列长度
    num_classes = num_classes #类别数
    num_filters = 256  #卷积核数目
    kernel_size = 3  #卷积核尺寸
    vocab_size = 5000  #字符表大小
    hidden_dim = 128  #全连接层神经元
    dropout_keep_prob = 0.5  #dropout比例
    learning_rate = 1e-3  #学习率
    batch_size = 64
    num_epochs = 7
    print_per_batch = 20
    save_per_batch = 10

class TextCNN(object):

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')#输入
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        #字向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')#卷积层
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')#池化层

        with tf.name_scope("score"):
            #全连接层
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')# 分类器
            self.y_pred_cls1 = tf.sigmoid(self.logits)#（0-1）
            self.y_pred_cls = tf.sigmoid(self.logits)
            one = tf.ones_like(self.logits)
            zero = tf.zeros_like(self.logits)
            self.y_pred_cls = tf.where(self.y_pred_cls <0.5, x=zero, y=one)
            self.y_pred_cls_one = tf.argmax(tf.nn.softmax(self.logits), 1)
            self.input_y_1=tf.argmax(tf.nn.softmax(self.input_y),1)
        with tf.name_scope("optimize"):
            mutli_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)#交叉熵
            self.loss = tf.reduce_mean(mutli_label_loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(self.input_y, self.y_pred_cls)
            correct_pred1 = tf.equal(self.input_y_1, self.y_pred_cls_one)
            print(correct_pred.shape)
            correct_pred = tf.reduce_sum(tf.cast(correct_pred, tf.float32), 1)
            print(correct_pred.shape)
            one = tf.ones_like(self.y_pred_cls_one)
            zero = tf.zeros_like(self.y_pred_cls_one)
            correct_pred = tf.where(correct_pred < 6, x=zero, y=one)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            self.acc1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
