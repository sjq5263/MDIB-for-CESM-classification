import numpy as np
import os
import os.path as p
from glob import glob
import errno


CLASSES = './classes.txt'
LISTS_TRAIN = './list/ttrain/'
LISTS_TEST =  './list/ttest/'
IMAGES = './data/' 

classes = np.loadtxt(CLASSES, dtype=str)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def gen_list(listdir, imagedir, clz): # (listdir_train, imagedir_train, cind)
    models = os.listdir(imagedir)
    for m in models: 
        imfiles = glob(p.join(imagedir, m, '*.jpg'))
        imfiles = sorted(imfiles, key=lambda s:(s.split('.')[-2]))
        listfile = p.join(listdir, '%s.txt' % m)
        
        with open(listfile, 'w+') as f:
            
            #print>>f, clz # yuan dai ma label
            print(clz, file=f) # sjq diyici xiugai label
            #f.write('clz')  # sjq dierci xiugai label
            #print>>f, 12 # yuan dai ma 12 views
            print(4, file=f) # sjq diyici xiugai 12 views
            #f.write('12') # sjq dierci xiugai label
            for imfile in imfiles:
                #print>>f, imfile  yuan dai ma 
                print(imfile, file=f)
                #f.write('imfile') # sjq dierci xiugai label


for cind, c in enumerate(classes):
    listdir_train = LISTS_TRAIN + c  # LISTS_TRAIN = './list/train/'
    listdir_test = LISTS_TEST + c   # LISTS_TEST =  './list/test/'
    imagedir_train = IMAGES + c + '/train/'  #IMAGES = './data2/'
    imagedir_test = IMAGES + c + '/test/'
    
    mkdir_p(LISTS_TRAIN + c)
    mkdir_p(LISTS_TEST + c)
    
    gen_list(listdir_train, imagedir_train, cind)
    gen_list(listdir_test, imagedir_test, cind)


