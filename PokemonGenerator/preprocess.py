
import numpy as np
import cv2
import pickle
import os
import time

import tensorflow as tf


CHANNEL = 1
DATA_FILE = 'data/trainingSample'
SAVE_FILE = 'Datav1.4'

class preprocess:

    def __init__(self):
        self.X = []

    def load(self):
        picklefile = open('pickled/'+SAVE_FILE,'r')
        obj = pickle.loads(picklefile.read())
        return obj

    def saveData(self):

        current_dir = os.getcwd()
        # parent = os.path.dirname(current_dir)
        pokemon_dir = os.path.join(current_dir, DATA_FILE)
        images = []
        for each in os.listdir(pokemon_dir):
            images.append(os.path.join(pokemon_dir,each))

        all_images = []
        decoded_files = []
        sess = tf.Session()

        for im in images:
            content = tf.read_file(im)
            img = tf.image.decode_jpeg(content, channels = CHANNEL)
            decoded_files.append(img)

        counter = 0
        for i in range(0,len(decoded_files),5000):

            img = tf.to_float(tf.image.resize_images(decoded_files[i:i+5000], [32,32], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
            img =  sess.run(img)
            counter += 1
            if counter%5 == 0:
                print i


        self.X = np.array(img)



if __name__ == '__main__':
    start = time.time()
    obj = preprocess()
    obj.saveData()
    pickle_file_sampled_data = open('pickled/'+SAVE_FILE,'w')
    pickle.dump(obj,pickle_file_sampled_data)
    print "seconds ---------- "+str(time.time()-start)
