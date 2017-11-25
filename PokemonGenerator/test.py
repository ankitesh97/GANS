
import tensorflow as tf

def a(resuse=False):
    with tf.variable_scope('a') as scope:
        if reuse:
            scope.reuse_variables()

        w_a = tf.get_variable('w_a',shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        b_a = tf.get_variable('b_a',shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(2.0))

        a_op = tf.multiply(w_a,b_a)
        tf.assign(w_a,[20.0])
        tf.assign(b_a,[21.0])

        return a_op



def b(reuse=False):
    with tf.variable_scope('b') as scope:
        if reuse:
            scope.reuse_variables()

        w_b = tf.get_variable('w_b',shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(3.0))
        b_b = tf.get_variable('b_b',shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(4.0))

        b_op = tf.add(w_b,b_b, name='final')
        r = tf.assign(w_b,[10.0])
        j = tf.assign(b_b,[11.0])

        return b_op,r,j



def train():
    b_ops, r, j = b()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print sess.run(b_ops)
        print sess.run(r)
        print sess.run(j)
        print sess.run(b_ops)
        saver.save(sess,'model')


def test():

    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('.'))
        # b_ops, _, _ = b(False)
        # sess.run(tf.global_variables_initializer())

        print sess.run('b/final:0')

def check(reuse = False):

    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        w2 = tf.get_variable('w2', shape=[10, 1], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))
    return w2


def main():
    w_op  = check()
    w_op2  = check(reuse=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print sess.run(w_op)
        print sess.run(w_op2)

# train()
# test()
main()
