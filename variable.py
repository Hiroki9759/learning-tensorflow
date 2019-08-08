import tensorflow as tf

#変数の定義
counter = tf.Variable(0,name = "counter")
step_size = tf.constant(1,name = "step_size")

#現在のCOUNTERにstep_sizeを足す
increment_op = tf.add(counter,step_size)
#increment_op の演算結果でcounterの値を更新
count_up_op = tf.assign(counter,increment_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        print(sess.run(count_up_op))
        