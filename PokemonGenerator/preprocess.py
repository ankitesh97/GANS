
import numpy as np
import cv2
import pickle
import os
import time

DATA_FILE = 'data/In_series'
SAVE_FILE = 'Datav0.0'

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
        for im in images:
            img = cv2.imread(im, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(128,128))
            img = (img*1.0)/255
            all_images.append(img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        self.X = np.array(all_images)



if __name__ == '__main__':
    start = time.time()
    obj = preprocess()
    obj.saveData()
    pickle_file_sampled_data = open('pickled/'+SAVE_FILE,'w')
    pickle.dump(obj,pickle_file_sampled_data)
    print "seconds ---------- "+str(time.time()-start)
