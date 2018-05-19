from __future__ import division

import numpy as np
import scipy as sp
from scipy.io import loadmat
from matplotlib import pyplot as plt
import matplotlib as mpl
import csv
import sys, ast
import pdb
csv.field_size_limit(sys.maxsize)

from wyrm import processing as proc
from wyrm.types import Data
from wyrm import plot
from wyrm.io import load_bcicomp3_ds1
plot.beautify()

DATA_DIR = 'data/'
TRUE_LABELS = 'data/true_labels.txt'

# load test and training data

train_instances = []
train_labels = []
test_instances = []
test_labels = []

def read_csv(name, target):
    with open(name, "rb") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            row = eval(row[0])
            target.append(row)

# read training and test data
for i in range(1,2):
    print "starting " + str(i)
    # training data:
    read_csv(DATA_DIR + "traindata/trial" + str(i) + "instances.csv", train_instances)
    read_csv(DATA_DIR + "traindata/trial" + str(i) + "_labels.csv", train_labels)
    # test data:
    #read_csv(DATA_DIR + "evaluationdata/trial" + str(i) + "instances.csv", test_instances)
    #read_csv(DATA_DIR + "evaluationdata/trial" + str(i) + "_labels.csv", test_labels)

pdb.set_trace()

# convert labels into ints
temp = np.array(train_labels, dtype = np.int8)

sorted_by_labels = [[], [], []]
for i, label in enumerate(train_labels):
    sorted_by_labels[int(label[0])].append(train_instances[i])
pdb.set_trace()
shape = (2, len(sorted_by_labels[1][0]), len(sorted_by_labels[1][0]))
csp_training_data = np.ndarray(sorted_by_labels[1:2]) # only want positive classes in our CSP training data

test_data = Data()

dat_train, dat_test = load_bcicomp3_ds1(DATA_DIR)
pdb.set_trace()

# load true labels
true_labels = np.loadtxt(TRUE_LABELS).astype('int')
# map labels -1 -> 0
true_labels[true_labels == -1] = 0
def plot_csp_pattern(a):
    # get symmetric min/max values for the color bar from first and last column of the pattern
    maxv = np.max(np.abs(a[:, [0, -1]]))
    minv = -maxv

    im_args = {'interpolation' : 'None', 
        'vmin' : minv, 
        'vmax' : maxv
    }

    # plot
    ax1 = plt.subplot2grid((1,11), (0,0), colspan=5)
    ax2 = plt.subplot2grid((1,11), (0,5), colspan=5)
    ax3 = plt.subplot2grid((1,11), (0,10))

    ax1.imshow(a[:, 0].reshape(8, 8), **im_args)
    ax1.set_title('Pinky')

    ax = ax2.imshow(a[:, -1].reshape(8, 8), **im_args)
    ax2.set_title('Tongue')

    plt.colorbar(ax, cax=ax3)
    plt.tight_layout()

def preprocess(data, filt=None):
    dat = data.copy()
    fs_n = 250 # sample rate is 250 for us

    b, a = proc.signal.butter(5, [13 / fs_n], btype='low')
    dat = proc.filtfilt(dat, b, a)

    b, a = proc.signal.butter(5, [9 / fs_n], btype='high')
    dat = proc.filtfilt(dat, b, a)

    dat = proc.subsample(dat, 50)

    if filt is None:
        filt, pattern, _ = proc.calculate_csp(dat)
        plot_csp_pattern(pattern)
    dat = proc.apply_csp(dat, filt)

    dat = proc.variance(dat)
    dat = proc.logarithm(dat)
    return dat, filt

fv_train, filt = preprocess(dat_train)
fv_test, _ = preprocess(dat_test, filt)

cfy = proc.lda_train(fv_train)
result = proc.lda_apply(fv_test, cfy)
result = (np.sign(result) + 1) / 2
print 'LDA Accuracy %.2f%%' % ((result == true_labels).sum() / len(result))

plt.show()
