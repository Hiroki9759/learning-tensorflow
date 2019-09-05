from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow と keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("train_images.shape")
print(train_images.shape)
print("len(train_labels)")
print(len(train_labels))
print("train_labels")
print(train_labels)

print("test_images.shape")
print(test_images.shape)
print("len(test_labels)")
print(len(test_labels))
print("test_labels")
print(test_labels)

#データの前処理
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()


#スケーリング
train_images = train_images/255.0
test_images = test_images/255.0
#正しいフォーマットか確認
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    #28*28ピクセルの２次元配列から784ピクセルの１次元配列に変換
    keras.layers.Flatten(input_shape=(28,28)),
    #密結合あるいは全結合されたニューロンの層
    keras.layers.Dense(128, activation=tf.nn.relu),
    #１０ノードのsoftmax層で各クラスに属する確率を示す
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#モデルのコンパイル
model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#optimizer:モデルが見ているデータと損失関数の値からどのようにモデルを更新するか
#loss:どのくらいモデルが正確かを測定する
#metrics:訓練とテストのステップを監視するのに使用する。今回は画像が正しく分類された比率(accuracy)を使用する


#モデルの訓練
model.fit(train_images,train_labels,epochs=10)

#正解率の評価
test_loss, test_acc = model.evaluate(test_images,test_labels)
print('Test accuracy', test_acc)

#予測する
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))

print(test_labels[0])
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
# X個のテスト画像、予測されたラベル、正解ラベルを表示します。
# 正しい予測は青で、間違った予測は赤で表示しています。
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# テスト用データセットから画像を1枚取り出す
img = test_images[0]

print(img.shape)

# 画像を1枚だけのバッチのメンバーにする
img = (np.expand_dims(img,0))

print(img.shape)
predictions_single = model.predict(img)

print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
prediction = predictions[0]

print(np.argmax(prediction))