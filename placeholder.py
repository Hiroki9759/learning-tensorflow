import tensorflow as tf
#placeholderを使ってx,yを定義
x = tf.placeholder(tf.int32,name = 'x')
y = tf.placeholder(tf.int32,name = 'y')

#加算
add_op = tf.add(x,y)
#乗算
mul_op = tf.multiply(x,y)

with tf.Session() as sess:
    print(sess.run(add_op,feed_dict={x:1,y:2}))
    print(sess.run(mul_op,feed_dict={x:1,y:2}))

    print(sess.run(add_op,feed_dict={x:100,y:200}))
    print(sess.run(mul_op,feed_dict={x:100,y:200}))
