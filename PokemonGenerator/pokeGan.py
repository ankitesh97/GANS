
import numpy as np
import tensorflow as tf
from preprocess import preprocess
import os
import sys
import cv2
import scipy

HEIGHT, WIDTH, CHANNEL = 128,128,3
BATCH_SIZE = 64
EPOCHS = 10
SAMPLE_SIZE = 16
random_dim = 100
learning_rate = 0.00005

generate_noise = np.random.normal(size=[BATCH_SIZE, random_dim]).astype(np.float32)

def lrelu(x, leak=0.3, n="lrelu"):
    with tf.variable_scope(n):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)



def to_range(images, min_value=0.0, max_value=1.0, dtype=None):

    return (images * 255.).astype(dtype=np.int16) % 256

def to_range_scipy(image):
    return (image+1.)/2.

def imwrite(image, path):

    if image.ndim == 3 and image.shape[2] == 1:
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    cv2.imwrite(path,to_range(image,0,255))
    imwrite_scipy(image,'scipy/'+path)

def imwrite_scipy(image,path):
    scipy.misc.imsave(path, to_range_scipy(image))

def merge_images(images, size):
    # images is of shape (?xhxwxc)
    # size is (#rows,#cols for the merged image)
    h,w,c = images.shape[1:]
    merged = np.zeros((h*size[0], w*size[1],3))
    for idx, image in enumerate(images):
        i = idx%size[1]
        j = idx//size[1]
        merged[j*h:j*h+h, i*w:i*w+w,:] = image
    return merged


def generator(input, random_dim, is_train=True, reuse=False):
    c4, c8, c16, c32, c64 = 128, 64, 32, 16,8
    s4 = 4
    output_dim = 3
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.layers.batch_normalization(conv1, training=is_train, name='bn1')
        act1 = tf.nn.relu(bn1, name='act1')

        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.layers.batch_normalization(conv2, training=is_train, name='bn2')
        act2 = tf.nn.relu(bn2, name='act2')

        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, training=is_train, name='bn3')
        act3 = tf.nn.relu(bn3, name='act3')

        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.layers.batch_normalization(conv4, training=is_train, name='bn4')
        act4 = tf.nn.relu(bn4, name='act4')

        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv5')
        bn5 = tf.layers.batch_normalization(conv5, training=is_train, name='bn5')
        act5 = tf.nn.relu(bn5, name='act5')

        conv6 = tf.layers.conv2d_transpose(act5, c64, kernel_size=[6, 6], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv6')
        bn6 = tf.layers.batch_normalization(conv6, training=is_train, name='bn6')
        act6 = tf.nn.relu(bn6, name='act6')

        conv7 = tf.layers.conv2d_transpose(act6, output_dim, kernel_size=[128, 128], strides=[1, 1], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv7')
        act7 = tf.nn.tanh(conv7, name='act7')

        return act7


def discriminator(input, is_train=True, reuse=False):
    c2, c4, c8, c16 = 16, 32, 64, 128
    with tf.variable_scope('disc') as scope:
        if reuse:
            scope.reuse_variables()

        conv1 = tf.layers.conv2d(input, c2, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        act1 = lrelu(conv1, n='act1')

        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.layers.batch_normalization(conv2, training=is_train, name='bn2')
        act2 = lrelu(bn2, n='act2')

        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, training=is_train, name='bn3')
        act3 = lrelu(bn3, n='act3')

        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.layers.batch_normalization(conv4, training=is_train, name='bn4')
        act4 = lrelu(bn4, n='act4')


        conv5 = tf.layers.conv2d(act4, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv5')
        bn5 = tf.layers.batch_normalization(conv5, training=is_train, name='bn5')
        act5 = lrelu(bn5, n='act5')

        shape = act5.get_shape().as_list()
        dim = shape[1] * shape[2] * shape[3]
        fc1 = tf.reshape(act5, shape=[-1, dim], name='fc1')
        w1 = tf.get_variable('w1', shape=[fc1.shape[1], 16], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[16], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        output = tf.add(tf.matmul(fc1, w1), b1, name='output')
        return output




def getNextBatch(X,pos,BATCH_SIZE):
    if pos+BATCH_SIZE > len(X):
        np.random.shuffle(X)
        pos = 0

    return X[pos:pos+BATCH_SIZE], pos+BATCH_SIZE

def train():
    X = preprocess().load().X
    m = X.shape[0]
    d_iters = 5
    gLosses = []
    dLosses = []

    real_image = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNEL],name='image_input')
    random_inp = tf.placeholder(dtype=tf.float32,shape=[None,random_dim],name='random_inp')

    fake_image_generator = generator(random_inp, random_dim, reuse=False)
    real_result = discriminator(real_image)
    fake_result = discriminator(fake_image_generator,reuse=True)
    sample_img = generator(random_inp,random_dim,is_train=True,reuse=True)

    # loss functions
    d_loss = tf.reduce_mean(real_result-fake_result)
    g_loss =  tf.reduce_mean(fake_result)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'disc' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]

    trainer_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_vars)
    d_clip = [v.assign(tf.clip_by_value(v, -0.01,0.01)) for v in d_vars]

    tf.summary.scalar('Loss_Discriminator',d_loss)
    tf.summary.scalar('Loss_Generator',g_loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)
    sess = tf.Session()
    sess.run(init)
    # saver.restore(sess,'./models/PokeGanModelV1/model-2000')
    writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
    write_op = tf.summary.merge_all()
    print "-----------------------------------Starting to train---------------------------------"
    sys.stdout.flush()
    pos = 0
    total_batch = 0
    for epoch in range(EPOCHS):

        for batch in range(X.shape[0]/BATCH_SIZE):
            d_iters = 5
            if total_batch%500==0 or total_batch < 25:
                d_iters = 5
            ## discriminator loop
            for d_n in range(d_iters):
                curr_batch, pos = getNextBatch(X,pos,BATCH_SIZE)
                train_noise = np.random.normal(size=[BATCH_SIZE, random_dim]).astype(np.float32)
                #train the desc
                sess.run(d_clip)
                _, dLoss = sess.run([trainer_d, d_loss], feed_dict={real_image:curr_batch,random_inp:train_noise})


            train_noise = np.random.normal(size=[BATCH_SIZE, random_dim]).astype(np.float32)
            _, gLoss = sess.run([trainer_g, g_loss], feed_dict={random_inp:train_noise})

            total_batch += 1

            #get the summary
            summary = sess.run([write_op] ,feed_dict={random_inp:train_noise,real_image:curr_batch})[0]
            writer.add_summary(summary,total_batch)
        # save the model
        if epoch%5==0:
            select = np.random.randint(0,X.shape[0],BATCH_SIZE)
            curr_batch = X[select]
            train_noise = np.random.normal(size=[BATCH_SIZE, random_dim]).astype(np.float32)
            dLoss = sess.run(d_loss, feed_dict={real_image:curr_batch,random_inp:train_noise})
            train_noise = np.random.normal(size=[BATCH_SIZE, random_dim]).astype(np.float32)
            gLoss = sess.run(g_loss,feed_dict={random_inp:train_noise})
            print "EPOCH: "+str(epoch)+" Disc Loss: "+str(dLoss)+" Gen Loss: "+str(gLoss)
            # save some images
            if epoch%100 == 0:
                os.makedirs('generated/'+str(epoch)+'/')
                os.makedirs("scipy/generated/"+str(epoch)+'/')

                gen_images = sess.run(sample_img,feed_dict={random_inp:generate_noise})
                for g in range(len(gen_images)):
                    imwrite(gen_images[g],'generated/'+str(epoch)+"/"+str(g)+'.jpg')
                train_noise = np.random.normal(size=[BATCH_SIZE, random_dim]).astype(np.float32)
                gen_images = sess.run(sample_img,feed_dict={random_inp:train_noise})
                os.makedirs('generated/'+str(epoch)+'/random/')
                os.makedirs("scipy/generated/"+str(epoch)+'/random/')

                merged = merge_images(gen_images,(8,8))
                imwrite(merged,'generated/'+str(epoch)+'/random/'+str(g)+'.jpg')

        if(epoch%200==0):
            saver.save(sess,'models/PokeGanModelV1/model', global_step=epoch, write_meta_graph=False)
            with open("control.txt",'r') as f:
                control = f.read()
                if control.strip() == "1":
                   print "stopping the training process .........."
                   sys.stdout.flush()
                   break



    saver.save(sess,'models/PokeGanModelV1/model', global_step=EPOCHS)

train()
