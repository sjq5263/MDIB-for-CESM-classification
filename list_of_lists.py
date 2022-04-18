# generate train_lists.txt, val_lists.txt, test_lists.txt
# each including lines of the lists (the ones in list/) of views
import numpy as np
from glob import glob
import os.path as p
import re
import shutil
import os

LISTS = './list/'
TRAIN = LISTS + 'ttrain/'
TEST = LISTS + 'ttest/'
CLASSES = './classes.txt'

TRAIN_OUT = './ttrain_lists.txt'
TEST_OUT = './ttest_lists.txt'
VAL_OUT = './ttval_lists.txt'


def out_lists(outfile, lists, class_index):
    with open(outfile, 'a+') as f:
        for l in lists:
           #print>>f, '%s %d' % (l, class_index)
           #print(f, '%s %d' % (l, class_index)) # sjq diyici xiugai 
           #f.write('%s %d' % (l, class_index))  # sjq dierci xiugai
           print('%s %d' % (l, class_index), file=f) # sjq disanci xiugai


def clear_outfiles():
    try:
        os.remove(TRAIN_OUT)
    except:
        pass
    try:
        os.remove(VAL_OUT)
    except:
        pass
    try:
        os.remove(TEST_OUT)
    except:
        pass

clear_outfiles()

classes = np.loadtxt(CLASSES, dtype=str)
for c_index, c in enumerate(classes):
    lists = glob(p.join(TRAIN, c, '*.txt')) # lists= './list/train/cancer/58.txt'
    def get_id_of_list(l):
        try:
            id_ = re.split('.', p.basename(l)[0])
            return id_
        except:
            #print l
            print(l)

    lists = sorted(lists, key=get_id_of_list)

    
    # train/val
    train_lists = lists[:-47]
    val_lists = lists[-47:]

    out_lists(TRAIN_OUT, train_lists, c_index)
    out_lists(VAL_OUT, val_lists, c_index)

    
    test_lists = glob(p.join(TEST, c, '*.txt'))
    test_lists = sorted(test_lists, key=get_id_of_list)
    out_lists(TEST_OUT, test_lists, c_index)
