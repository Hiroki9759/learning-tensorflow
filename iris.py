
import numpy as np
import tensorflow as tf 
import urllib.request


#データセットの取得
IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

#データを取得してCSVファイルとして書き出す
with open(IRIS_TRAINING,'wb') as f:
    f.write(urllib.request.urlopen(IRIS_TRAINING_URL).read())
with open(IRIS_TEST,'wb') as f:
    f.write(urllib.request.urlopen(IRIS_TEST_URL).read())

#データの整形
training_data = np.loadtxt('./iris_training.csv',delimiter=',',skiprows=1)
train_x = training_data[:,:-1]
train_y = training_data[:,-1]
test_data = np.loadtxt('./iris_test.csv',delimiter=',',skiprows=1)
test_x = test_data[:,:-1]
test_y = test_data[:,-1]

#データの中から、指定されたバッチサイズのぶんだけランダムに取得
def next_batch(data,label,batch_size):
    indices = np.random.randint(data.shape[0],size=batch_size)
    return data[indices],label[indices]

#ニューラルネットワークの構築
##入力層の構築
x = tf.placeholder(tf.float32,[None,4],name = 'input')

##中間層１
hidden1 = tf.layers.dense(inputs = x, units = 10, activation = tf.nn.relu, name = "hidden1" )

##中間層２
hidden2 = tf.layers.dense(inputs = hidden1, units = 20, activation = tf.nn.relu, name = "hidden2" )
##中間層３
hidden3 = tf.layers.dense(inputs = hidden2, units = 10, activation = tf.nn.relu )

##出力層
y = tf.layers.dense(inputs = hidden3, units = 3, activation = tf.nn.softmax, name = "output" )
# model = tf.keras.Sequential()
# model.add(tf)
# y = hidden3.add(tf.keras.layers.Dense(3,activation=tf.nn.softmax))
##実際のあやめの種類
labels = tf.placeholder(tf.int64,[None],name = 'teacher_signal')
y_ = tf.one_hot(labels,depth=3,dtype = tf.float32)
##実際のあやめの種類とネットワークからの出力を比較
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

##学習結果の評価
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#学習の実行
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

batch_size =20
epoch = 1500
batch_in_epoch = training_data.shape[0] //batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        for j in range(batch_in_epoch):
            #学習処理
            batch_train_x,batch_train_y = next_batch(train_x,train_y,batch_size)
            sess.run(train_op,feed_dict={x:batch_train_x,labels:batch_train_y})
        print(sess.run(accuracy,feed_dict={x:test_x,labels:test_y}))