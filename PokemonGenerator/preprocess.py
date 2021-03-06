
import numpy as np
import cv2
import pickle
import os
import json
import time

import tensorflow as tf


CHANNEL = 3
size = [128,128]
DATA_FILE = 'data/rotated'
LOAD_FILE = 'pickled/dataPokemonAugmented.npy'

class preprocess:

    def __init__(self):
        self.X = []

    def load(self,read=True):
        save_file = open(LOAD_FILE,'r')
        self.X = np.load(save_file)["X"]
        if read==False:
            obj = self
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
        print len(decoded_files)
        img2 = []

        for i in range(0,len(decoded_files),1000):

            img = tf.to_float(tf.image.resize_images(decoded_files[i:i+1000], size, method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
            img =  sess.run(img)
            for j in img:
                j = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
                #cv2.imshow("img",j)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                img2.append(j)
            counter += 1
            if counter%5 == 0:
                print i

        save_file = open(LOAD_FILE,'w')
        self.X = np.array(img2)
        np.savez(save_file,X=self.X)


if __name__ == '__main__':
    start = time.time()
    obj = preprocess()
    obj.saveData()
    print "seconds ---------- "+str(time.time()-start)
