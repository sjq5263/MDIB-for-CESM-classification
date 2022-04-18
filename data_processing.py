
import random
import numpy as np
import time
import queue
import threading

from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2

W = H = 227

class Shape:
    def __init__(self, list_file):
        with open(list_file) as f:
            self.label = int(f.readline())
            self.V = int(f.readline())
            view_files = [l.strip() for l in f.readlines()]
        # a list of views for a single object
        self.views = self._load_views(view_files, self.V)
        self.done_mean = False


    def _load_views(self, view_files, V):
        views = []
        for f in view_files:
            im = cv2.imread(f)
            im = cv2.resize(im, (W, H))
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
            # assert im.shape == (W,H,3), 'BGR!'
            im = im.astype('float32')
            views.append(im)
        views = np.asarray(views)
        return views

    def subtract_mean(self):
        self.views -= np.mean(self.views)
        self.views /= np.std(self.views)
        ##if not self.done_mean:
          ##  mean_bgr = (104., 116., 122.)
            ##for i in range(3):
              ##  self.views[:,:,:,i] -= mean_bgr[i]

        self.done_mean = True

    def crop_center(self, size=(227,227)):
        w, h = self.views.shape[1], self.views.shape[2]
        wn, hn = size
        #left = w / 2 - wn / 2  yuan dai ma 
        #top = h / 2 - hn / 2   yuan  dai ma
        left = w // 2 - wn // 2
        top = h // 2 - hn // 2
        right = left + wn
        bottom = top + hn
        self.views = self.views[:, left:right, top:bottom, :]

class Dataset:
    def __init__(self, listfiles, labels, subtract_mean, V):  
        self.listfiles = listfiles  # listfiles_train = ./list/train/normal/36.txt
        self.labels = labels
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.V = V

    def size(self):
        return len(self.listfiles) 

    def shuffle(self):
        # z = zip(self.listfiles, self.labels) # yuandaima
        z = list(zip(self.listfiles, self.labels))
        random.shuffle(z)
        self.listfiles, self.labels = [list(l) for l in zip(*z)]
        self.shuffled = True

    def load_shape(self, views_for_a_single_object):
        s = Shape(views_for_a_single_object)
        # s.crop_center()    
        if self.subtract_mean:
            s.subtract_mean()
        return s

    def batches(self, batch_size):
        listfiles = self.listfiles
        n = len(listfiles)  # train_listfiles = 130
        # for i in xrange(0, n, batch_size): #yuan dai ma
        for i in range(0, n, batch_size):
            lists = listfiles[i: i + batch_size]
            # x = np.zeros((batch_size, self.V, 227, 227, 3))
            x = np.zeros((batch_size, self.V, 227, 227, 3))
            y = np.zeros(batch_size)

            for index, file in enumerate(lists):
                shape_object = self.load_shape(file)
                x[index, ...] = shape_object.views
                y[index] = shape_object.label

            yield x, y

    def sample_batches(self, batch_size, n):
        listfiles = random.sample(self.listfiles, n)
        n = len(listfiles)
        for i in xrange(0, n, batch_size):
            lists = listfiles[i: i + batch_size]
            #x = np.zeros((batch_size, self.V, 227, 227, 3))
            x = np.zeros((batch_size, self.V, 227, 227, 3))
            y = np.zeros(batch_size)

            for index, file in enumerate(lists):
                shape_object = self.load_shape(file)
                x[index, ...] = shape_object.views
                y[index] = shape_object.label

            yield x, y
