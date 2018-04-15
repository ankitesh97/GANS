
import numpy as np
import tensorflow as tf
from preprocess import preprocess
import os
import sys
import cv2
import scipy
import argparse

HEIGHT, WIDTH, CHANNEL = 64,64,3
BATCH_SIZE = 64
EPOCHS = 200
SAMPLE_SIZE = 16
random_dim = 100
learning_rate = 2e-4

generate_noise = np.random.uniform(-1.0,1.0,size=[BATCH_SIZE, random_dim]).astype(np.float32)

MODEL_ADD = 'models/mnist/'
parser = argparse.ArgumentParser()
parser.add_argument("--test",help="Generates the number of images specified",type=int)
def lrelu(x, leak=0.3, n="lrelu"):
    with tf.variable_scope(n):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)



def to_range(images, min_value=0.0, max_value=1.0, dtype=None):

    return (images * 255.).astype(dtype=np.int16) % 256

def to_range_scipy(image):
    return (image+1.)/2.

def imwrite(image, path,write_scipy=True):

    if image.ndim == 3 and image.shape[2] == 1:
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    # cv2.imwrite(path,to_range(image,0,255))
    if write_scipy:
        imwrite_scipy(image,path)

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



def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 128, 64, 32, 16,8  # channel num,256,128,64,32
    s4 = 4
    output_dim = 3  # gray image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # 4*4*256
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, scope='bn1', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*64
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, scope='bn2', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, scope='bn3', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*256
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, scope='bn4', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        act4 = tf.nn.relu(bn4, name='act4')

        conv5 = tf.layers.conv2d_transpose(act4, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv5')
        # bn5 = tf.contrib.layers.batch_norm(conv5, scope='bn5', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        # act5 = tf.nn.relu(bn5, name='act5')
        # 32*32*1
        # conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[64, 64], strides=[1, 1], padding="SAME",
                                           # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           # name='conv6')
        act6 = tf.nn.tanh(conv5, name='act6')

        return act6



def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 16, 32, 64, 128  # channel num,32, 64, 128
    with tf.variable_scope('disc') as scope:
        if reuse:
            scope.reuse_variables()
        # 16*16*32
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        act1 = lrelu(conv1, n='act1')

        # 8*8*64
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, scope='bn2', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        act2 = lrelu(bn2, n='act2')
        # 4*4*128
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, scope='bn3', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        act3 = lrelu(bn3, n='act3')

        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, scope='bn4', is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
        act4 = lrelu(bn4, n='act4')

        shape = act4.get_shape().as_list()
        dim = shape[1] * shape[2] * shape[3]
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
        w1 = tf.get_variable('w1', shape=[fc1.shape[1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        output = tf.add(tf.matmul(fc1, w1), b1, name='output')
        return output




def getNextBatch(X,pos,BATCH_SIZE):
    if pos+BATCH_SIZE > len(X):
        np.random.shuffle(X)
        pos = 0

    return X[pos:pos+BATCH_SIZE], pos+BATCH_SIZE

def train():
    X = preprocess().load(read=False).X
    m = X.shape
    d_iters = 5
    gLosses = []
    dLosses = []
    print m
    is_train = tf.placeholder(tf.bool, name='is_train')
    real_image = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNEL],name='image_input')
    random_inp = tf.placeholder(dtype=tf.float32,shape=[None,random_dim],name='random_inp')

    fake_image_generator = generator(random_inp, random_dim,is_train, reuse=False)
    real_result = discriminator(real_image,is_train)
    fake_result = discriminator(fake_image_generator,is_train,reuse=True)
    sample_img = generator(random_inp,random_dim,is_train,reuse=True)

    # loss functions
    d_loss = tf.reduce_mean(fake_result)- tf.reduce_mean(real_result)
    g_loss =  -tf.reduce_mean(fake_result)

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
                d_iters = 25
            ## discriminator loop
            for d_n in range(d_iters):
                curr_batch, pos = getNextBatch(X,pos,BATCH_SIZE)
                train_noise = np.random.uniform(-1.0,1.0,size=[BATCH_SIZE, random_dim]).astype(np.float32)
                #train the desc
                sess.run(d_clip)
                _, dLoss = sess.run([trainer_d, d_loss], feed_dict={real_image:curr_batch,random_inp:train_noise,is_train:True})


            train_noise = np.random.uniform(-1.0,1.0,size=[BATCH_SIZE, random_dim]).astype(np.float32)
            _, gLoss = sess.run([trainer_g, g_loss], feed_dict={random_inp:train_noise,real_image:curr_batch,is_train:True})

            total_batch += 1

            #get the summary
            summary = sess.run([write_op] ,feed_dict={random_inp:train_noise,real_image:curr_batch,is_train:True})[0]
            writer.add_summary(summary,total_batch)
        # save the model
        if epoch%5==0:
            select = np.random.randint(0,X.shape[0],BATCH_SIZE)
            curr_batch = X[select]
            train_noise = np.random.uniform(-1.0,1.0,size=[BATCH_SIZE, random_dim]).astype(np.float32)
            dLoss = sess.run(d_loss, feed_dict={real_image:curr_batch,random_inp:train_noise,is_train:True})
            train_noise = np.random.uniform(-1.0,1.0,size=[BATCH_SIZE, random_dim]).astype(np.float32)
            gLoss = sess.run(g_loss,feed_dict={random_inp:train_noise,is_train:True})
            print "EPOCH: "+str(epoch)+" Disc Loss: "+str(dLoss)+" Gen Loss: "+str(gLoss)
            # save some images
            if epoch%5 == 0:
                os.makedirs('generated/'+str(epoch)+'/')
                os.makedirs("scipy/generated/"+str(epoch)+'/')

                gen_images = sess.run(sample_img,feed_dict={random_inp:generate_noise,is_train:False})
                for g in range(len(gen_images)):
                    imwrite(gen_images[g],'generated/'+str(epoch)+"/"+str(g)+'.jpg')
                train_noise = np.random.uniform(-1.0,1.0,size=[BATCH_SIZE, random_dim]).astype(np.float32)
                gen_images = sess.run(sample_img,feed_dict={random_inp:train_noise,is_train:False})
                os.makedirs('generated/'+str(epoch)+'/random/')
                os.makedirs("scipy/generated/"+str(epoch)+'/random/')

                merged = merge_images(gen_images,(8,8))
                imwrite(merged,'generated/'+str(epoch)+'/random/'+str(g)+'.jpg')

        if(epoch%200==0):
            saver.save(sess,MODEL_ADD+'model', global_step=epoch, write_meta_graph=False)
            with open("control.txt",'r') as f:
                control = f.read()
                if control.strip() == "1":
                   print "stopping the training process .........."
                   sys.stdout.flush()
                   break



    saver.save(sess,MODEL_ADD+'model', global_step=EPOCHS)


def test(total):
    is_train = tf.placeholder(tf.bool, name='is_train')
    random_inp = tf.placeholder(dtype=tf.float32,shape=[None,random_dim],name='random_inp')
    fake_image_generator = generator(random_inp, random_dim,is_train, reuse=False)
    sample_img = generator(random_inp,random_dim, is_train,reuse=True)
    saver = tf.train.Saver(max_to_keep=4)

    sess = tf.Session()
    saver.restore(sess,MODEL_ADD+'model-'+str(EPOCHS))

    noise = np.random.uniform(-1.0,1.0,size=[total, random_dim]).astype(np.float32)
    gen_images = sess.run(sample_img,feed_dict={random_inp:noise,is_train:False})
    os.makedirs('test_generated/')
    for g in range(len(gen_images)):
        imwrite(gen_images[g],'test_generated/'+str(g)+'.jpg',write_scipy=True)


if __name__ == '__main__':

    args = parser.parse_args()
    if args.test:
        test(args.test)
    else:
        train()
