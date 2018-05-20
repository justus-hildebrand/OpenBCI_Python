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
    read_csv(DATA_DIR + "evaluationdata/trial" + str(i) + "instances.csv", test_instances)
    read_csv(DATA_DIR + "evaluationdata/trial" + str(i) + "_labels.csv", test_labels)

np_train_instances = np.ndarray((len(train_instances), len(train_instances[0]), len(train_instances[0][0])), dtype=float)
for i, instance in enumerate(train_instances):
    for j, timepoint in enumerate(instance):
        for k, channel in enumerate(timepoint):
            np_train_instances[i][j][k] = float(channel)
np_test_instances = np.ndarray((len(test_instances), len(test_instances[0]), len(test_instances[0][0])), dtype=float)
for i, instance in enumerate(test_instances):
    for j, timepoint in enumerate(instance):
        for k, channel in enumerate(timepoint):
            np_test_instances[i][j][k] = float(channel)

# convert labels into ints
train_labels = np.array(train_labels, dtype = np.int8)
test_labels = np.array(test_labels, dtype = np.int8)

#sorted_by_labels = [[], [], []]
#for i, label in enumerate(train_labels):
#    sorted_by_labels[int(label[0])].append(train_instances[i])
#pdb.set_trace()
#shape = (2, len(sorted_by_labels[1][0]), len(sorted_by_labels[1][0]))
#csp_training_data = np.ndarray(sorted_by_labels[1:2]) # only want positive classes in our CSP training data
train_data = Data(np_train_instances, [train_labels, range(0,np_train_instances.shape[1]), range(1,26)], ["class", "time", "channel"], ["#", "1s/250", "#"])
train_data.fs = 250
train_data.class_names = ["none", "left", "right"]
test_data = Data(np_test_instances, [test_labels, range(0,np_test_instances.shape[1]), range(1,26)], ["class", "time", "channel"], ["#", "1s/250", "#"])
test_data.fs = 250



#dat_train, dat_test = load_bcicomp3_ds1(DATA_DIR)
dat_train = train_data
dat_test = test_data
#pdb.set_trace()

# TODO: FILTER OUT LABELS OF NaN INSTANCES

# load true labels
#true_labels = np.loadtxt(TRUE_LABELS).astype('int')
true_labels = test_labels
pdb.set_trace()
dat_train.data = dat_train.data[~np.isnan(dat_train.data)]
print dat_train.data[dat_train.data == float('inf')]
print len(dat_train.data[np.isnan(dat_train.data)])
print (np.isnan(dat_train.data)).any()
# map labels -1 -> 0
#true_labels[true_labels == -1] = 0
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
    fs_n = dat.fs / 2

    b, a = proc.signal.butter(5, [13 / fs_n], btype='low')
    dat = proc.filtfilt(dat, b, a)

    b, a = proc.signal.butter(5, [9 / fs_n], btype='high')
    dat = proc.filtfilt(dat, b, a)

    dat = proc.subsample(dat, 50)

    if filt is None:
        filt, pattern, _ = proc.calculate_csp(dat, classes = [1, 2])
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
