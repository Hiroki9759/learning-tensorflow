import tensorflow as tf

t1 = tf.constant(1,name='Tensor1')
t2 = tf.constant(2,name='Tensor2')
#加算
add_op = tf.add(t1,t2)
#乗算
mul_op = tf.multiply(t1,t2)

with tf.Session() as sess:
    print(sess.run(add_op))
    print(sess.run(mul_op))
