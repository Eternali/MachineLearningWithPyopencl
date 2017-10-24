#!/usr/bin/python3

import numpy as np
import pyopencl as cl
from pyopencl import array
import matplotlib.pyplot as plt
import scipy.io as spio
import time
import os
import sys

'''
This is a vanilla deep neural network that uses pyopencl to accelerate
the learning of parameters to classify handwritten digits.

Author: Conrad Heidebrecht    v. 0.0.1
'''


SAVEFILE = '/var/log/theta.mat'

epochs = 10000
alpha = 0.3
tolerance = 0.3
lam = 1.0
epsilon_init = 0.12


##----  FILE INPUT/OUTPUT FUNCTIONS  ------

def usage ():
    print('''
        ./dnn_imgclass_cl01.py --train traindata.mat [ --test testdata.mat ]
    ''')
    sys.exit()
    quit()


def check_permissions ():
    if os.getuid():
        usage()


def read_file (filename, mode='r'):
    contents = []
    with open(filename, mode) as fname:
        for line in fname:
            contents.append(line)

    return contents


def write_file (filename, data, mode='w'):
    with open(filename, mode) as fname:
        fname.writelines(data)


def parse_args ():
    args = ['', '']
    argv = sys.argv
    try:
        for a, arg in enumerate(argv):
            if arg == '--train':
                args[0] = argv[a + 1]
            elif arg == '--test':
                args[1] == argv[a + 1]
            elif arg == '--help' or arg == '-h':
                usage()
    except IndexError:
        usage()

    return args if '' not in args[0] else usage()


def read_data (filename, form='mat')
    if form == 'mat':
        data = spio.loadmat(filename, squeeze_me=True)
        X = np.array([np.array(x).astype(np.float32) for x in data['X']]).astype(np.float32)
        # transform decimal label values to one-hot vectors
        tmp_y = np.array(data['y']).astype(np.float32)
        y = np.zeros((tmp_y.shape[0], output_size), np.float32)
        for i, c in enumerate(tmp_y):
            y[i, c-1] = 1
    elif form == 'csv':
        data = read_file(filename)

    return X, y


def load_theta (filename='', form='mat', options=[]):
    thetas = {}
    if filename:
        if form == 'mat':
            data = spio.loadmat(filename, squeeze_me=True)
            for key in data.keys():
                thetas[key] = np.array(data[key]).astype(np.float32)
    else:
        for l, layer in enumerate(options):
            exec('thetas[\'Theta%d\'] = np.random.rand(%d, %d).astype(np.float32) * 2 * epsilon_init - epsilon_init'
                % (l + 1, layer['height'], layer['width']))

    return thetas


def save_theta (filename=SAVEFILE, thetas):
    os.system('touch %s' % filename)
    spio.savemat(filename, thetas)


##----  MATHEMATICAL HELPER FUNCTIONS  ------

def sigmoid (z, deriv=False):
    pass


##----  MAIN FUNCTION STARTS  ------

def main ():



if __name__ == '__main__':
    main()
