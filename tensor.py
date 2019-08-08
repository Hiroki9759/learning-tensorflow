import tensorflow as tf
#配列の定義（単一の数字,配列,多次元配列）
t1 = tf.constant(1,name='Rank0')
t2 = tf.constant([1,2],name='Rank1')
t3 = tf.constant([[1,2],[3,4]],name='Rank2')

with tf.Session() as sess:
    #printで標準出力に出力
    print(sess.run(t1))
    print(sess.run(t2))
    print(sess.run(t3))