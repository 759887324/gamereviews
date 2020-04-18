import tensorflow as tf
from config import num_classes
#超参数
class TRNNConfig(object):
    embedding_dim = 64
    seq_length = 200
    num_classes = num_classes
    vocab_size = 5000
    num_layers= 2           #隐藏层层数
    hidden_dim = 128        #隐藏层神经元数
    rnn = 'lstm'             #核选择
    dropout_keep_prob = 0.8
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 21
    print_per_batch = 50
    save_per_batch = 10


class TextRNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn()

    def rnn(self):
        def lstm_cell():   #lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
        def gru_cell():  #gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): #dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        #字向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 最后时刻和第2层的LSTM或GRU的隐状态作为结果

        with tf.name_scope("score"):
            # 全连接层
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls1 = tf.sigmoid(self.logits)
            self.y_pred_cls = tf.sigmoid(self.logits)
            one = tf.ones_like(self.logits)
            zero = tf.zeros_like(self.logits)
            self.y_pred_cls = tf.where(self.y_pred_cls <0.5, x=zero, y=one)
            self.y_pred_cls_one = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            self.input_y_1=tf.argmax(tf.nn.softmax(self.input_y),1)
        with tf.name_scope("optimize"):
            mutli_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)#交叉熵
            self.loss = tf.reduce_mean(mutli_label_loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
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