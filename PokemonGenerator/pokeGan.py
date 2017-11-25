
import numpy as np
import tensorflow as tf
from preprocess import preprocess
import os

HEIGHT, WIDTH, CHANNEL = 100, 100, 3
BATCH_SIZE = 32
EPOCHS = 100
SAMPLE_SIZE = 8


def getBatches(X, m):
    '''
    this function returns a list of tensors (which contains batches)
    '''
    np.random.shuffle(X)
    batches = []
    for batch in range(0,m,BATCH_SIZE):
        tf_batch = tf.convert_to_tensor(X[batch:batch+BATCH_SIZE], dtype=tf.float32)
        batches.append(tf_batch)
    return batches

def generator(input, random_dim, is_train):
    pass


def discriminator():
    pass



def train():
    X = preprocess().load().X
    m = X.shape[0]
    d_iters = 10
    gLosses = []
    dLosses = []
    saver = tf.train.Saver(max_to_keep=4)
    sess = tf.Session()
    for epoch in range(EPOCHS):
        batches = getBatches(X,m)
        n_batches = len(batches)

        for batch_num in range(n_batches):

            ## discriminator loop
            curr_batch = batches[batch_num]
            for d_n in range(d_iters):
                curr_sample = np.random.choice(curr_batch,SAMPLE_SIZE,replace=False)
                #random input
                train_noise = np.random.uniform(-1.0, 1.0, size=[SAMPLE_SIZE, random_dim]).astype(np.float32)
                #train the desc
                _, dLoss = sess.run([trainer_d, d_loss])
                dLosses.append(dLoss)

            train_noise = np.random.uniform(-1.0, 1.0, size=[SAMPLE_SIZE, random_dim]).astype(np.float32)
            _, gLoss = sess.run([trainer_g, g_loss])

            gLosses.append(gLoss)

        # save the model
        if(epoch%10==0):
            saver.save(sess,'PokeGanModel', global_step=epoch, write_meta_graph=False)

        if(epoch%5==0):
            #save some images
            os.makedirs('genrated/'+str(epochs))
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            gen_images = sess.run(fake_image)
            save_images(gen_images, epochs)

        with open("controlTraining.txt",'r') as f:
            control = f.read()
            if control.strip() == "1":
               print "stopping the training process .........."
               sys.stdout.flush()
               break

               
    saver.save(sess,'PokeGanModel', global_step=EPOCHS, write_meta_graph=False)
    os.makedirs('genrated/'+str(EPOCHS))
    sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
    gen_images = sess.run(fake_image)
    save_images(gen_images, epochs)

def test():
    pass

def save_images(genrated, epochs):
    pass

train()
