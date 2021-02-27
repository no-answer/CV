import tensorflow.compat.v1 as tf
import os
import pickle
import numpy as np

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0 # 占用GPU40%的显存
session = tf.Session(config=config)

tf.disable_v2_behavior()
cifar_dir = './cifar-10-batches-py'
print(os.listdir(cifar_dir))

# 构建文件路径
train_filenames = [os.path.join(cifar_dir, 'data_batch_%d'%i)for i in range(1,6)]
test_filenames = [os.path.join(cifar_dir, 'test_batch')]


# 数据载入函数
def load_data(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        return data[b'data'],data[b'labels']


# 数据处理类
class CifarData:
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)

        self._data = np.vstack(all_data) / 127.5 - 1
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._index = 0
        self._need_shuffle = need_shuffle
        if self._need_shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        o = np.random.permutation(self._num_examples)
        self._data = self._data[o]
        self._labels = self._labels[o]

    def next_batch(self,batch_size):
        end_index = self._index + batch_size
        if end_index > self._num_examples:
            if self._need_shuffle:
                self.shuffle_data()
                self._index = 0
                end_index = batch_size
            else:
                raise Exception('没有更多样本')
        if end_index > self._num_examples:
            raise Exception('尺寸过大')

        batch_data = self._data[self._index:end_index]
        batch_labels = self._labels[self._index:end_index]
        self._index = end_index
        return batch_data,batch_labels


# 实例化数据处理类
train_data = CifarData(train_filenames,True)
test_data = CifarData(test_filenames,False)

# 构建模型
X = tf.placeholder(dtype=tf.float32,shape=[None,3072])
Y = tf.placeholder(dtype=tf.int64,shape=[None])
X_img = tf.reshape(X,[-1,3,32,32])
X_img = tf.transpose(X_img,perm=[0,2,3,1])

# 构建神经网络
# 卷积一
conv1_1 = tf.layers.conv2d(X_img,8,kernel_size=(3,3),padding='same',activation=tf.nn.relu,name='conv1_1')
# 池化
pooling1 = tf.layers.max_pooling2d(conv1_1,(2,2),(2,2),name='pool1')

# 卷积二
conv2_1 = tf.layers.conv2d(pooling1,16,(3,3),padding='same',name='conv2_1',activation=tf.nn.relu)
# 池化
pooling2 = tf.layers.max_pooling2d(conv2_1,(2,2),(2,2),name='pool2')

# 卷积三
conv3_1 = tf.layers.conv2d(pooling2,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_2')
# 池化
pooling3 = tf.layers.max_pooling2d(conv3_2,(2,2),(2,2),name='pool3')

# 卷积四
conv4_1 = tf.layers.conv2d(pooling3,64,(3,3),padding='same',activation=tf.nn.relu,name='conv4_1')
conv4_2 = tf.layers.conv2d(conv4_1,64,(3,3),padding='same',activation=tf.nn.relu,name='conv4_2')
# 池化
pooling4 = tf.layers.max_pooling2d(conv4_2,(2,2),(2,2),name='pool4')

# 展平
flatten = tf.layers.flatten(pooling4,name='flaten')

# 全连接层
fc7 = tf.layers.dense(flatten,64,activation=tf.nn.tanh,name='fc7')
fc8 = tf.layers.dense(fc7,64,activation=tf.nn.tanh,name='fc8')

y_ = tf.layers.dense(fc8,10)

# 损失
loss = tf.losses.sparse_softmax_cross_entropy(labels=Y,logits=y_)

# 预测
predict = tf.argmax(y_,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype=tf.float32))

# 优化器
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 超参数
batch_size = 20
train_steps = 10000
test_steps = 100

# 开启会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_steps):
        x_train,y_train = train_data.next_batch(batch_size)
        los,acc,_ = sess.run([loss,accuracy,train_op],feed_dict={
            X:x_train,Y:y_train
        })
        if (i + 1) % 500 == 0:
            print('批次',i+1)
            print('代价:',los)
            print('准确率： ',acc)
        if (i + 1) % 5000 == 0:
            test_data = CifarData(test_filenames,False)
            all_acc = []
            for j in range(test_steps):
                x_test,y_test = test_data.next_batch(batch_size)
                accs = sess.run(accuracy,feed_dict={
                    X:x_test,Y:y_test
                })
                all_acc.append(accs)
            print('测试集准确率： ',sess.run(tf.reduce_mean(all_acc)))

