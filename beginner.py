#!/usr/bin/python3

import pyopencl as cl
from pyopencl import array
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as spio
import sys
import time


image_width = 20
image_height = 20

input_size = 400
hidden_size = 25
output_size = 10

epochs = 10000
alpha = 0.3
tolerance = 0.35
lam = 1.0
epsilon_init = 0.12


def check_permissions ():
    if os.getuid():
        usage()


def read_file (fname, mode='r'):
    contents = []
    with open(fname, mode) as f:
        for l in f:
            contents.append(l.strip())

    return contents


def write_file (fname, data, mode='w'):
    with open(fname, mode) as f:
        f.writelines(data)


def usage ():
    print('''
            ./beginner.py --train training_data training_labels [--test test_data test_labels] 
                          [--theta theta_file]
            
            ''')
    quit()


def parse_args ():
    args = ['', '', '', '', '']
    argv = sys.argv
    try: 
        for a, arg in enumerate(argv):
            if arg == '--train':
                args[:2] = argv[a+1], argv[a+2]
            elif arg == '--test':
                args[2:4] = argv[a+1], argv[a+2]
            elif arg == '--theta':
                args[4] = argv[a+1]
            elif arg == '-h' or arg == '--help':
                usage()
    except IndexError:
        usage()

    return args if '' not in args[:2] else usage()


def read_train (data_file, label_file, is_mat=True):
    if is_mat:
        data = spio.loadmat(data_file, squeeze_me=True)
        X = np.asarray(data['X'])
        tmp_y = np.asarray(spio.loadmat(label_file, squeeze_me=True)['y'])
        y = np.zeros((tmp_y.shape[0], output_size))
        for i, c in enumerate(tmp_y):
            y[i, c-1] = 1
        # or for demonstration
        # y = np.zeros((tmp_y.shape[0], output_size))
        # y[np.arange(tmp_y.shape[0]), tmp_y] = 1
    else:
        tmp_X = read_file(data_file)
        y = read_file(label_file)
        X = np.zeros((400, len(tmp_X)), np.float32)
        for i, x in enumerate(tmp_X):
            X[i] = tuple(x.split(','))

    return X, y


def load_theta (theta_file='', is_mat=True):
    if is_mat and theta_file:
        data = spio.loadmat(theta_file, squeeze_me=True)
        theta1 = np.asarray(data['Theta1'])
        theta2 = np.asarray(data['Theta2'])
    else:
        theta1 = np.random.rand(hidden_size, input_size + 1) * 2 * epsilon_init - epsilon_init
        theta2 = np.random.rand(output_size, hidden_size + 1) * 2 * epsilon_init - epsilon_init

    return theta1, theta2


def save_theta (theta_file, thetas):
    os.system('touch %s' % theta_file)
    spio.savemat(theta_file, thetas)


def show_image (x):
    reshaped = np.reshape(x, (20, 20))
    image = plt.imshow(reshaped, 'gray', origin='lower')
    plt.show()


# this will take one image and predict its digit
def predict (theta1, theta2, x):
    a1 = sigmoid(np.insert(x, 0, 1).dot(theta1.T))
    a2 = sigmoid(np.insert(a1, 0, 1).dot(theta2.T))
    guess_index = a2.argmax()

    return (guess_index + 1 if guess_index < a2.size - 1 else 0)


def get_correct (theta1, theta2, X, y):
    # forward propogate to get prediction
    a1 = sigmoid(np.insert(X, 0, 1, axis=1).dot(theta1.T))
    a2 = sigmoid(np.insert(a1, 0, 1, axis=1).dot(theta2.T))

    return np.sum(a2.argmax(axis=1) == y.argmax(axis=1))


def sigmoid (z, deriv=False):
    if not deriv:
        return 1.0 / (1.0 + np.exp(-z))
    return sigmoid(z) * (1 - sigmoid(z))


def softmax (z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


# return cost and gradients
# theta1 should be (rows, cols) 25, 401
# theta2 should be 10, 26
# X should be m, 400
# y should be m, 10
def update_weights (theta1, theta2, X, y, reg_lam):
    m = X.shape[0]
    # add bias
    a0 = np.insert(X, 0, 1, axis=1)  # m, 401
    # forward propagate to get activations
    z1 = a0.dot(theta1.T)  # m, 25
    a1 = np.insert(sigmoid(z1), 0, 1, axis=1)  # m, 26
    z2 = a1.dot(theta2.T)  # m, 10
    a2 = sigmoid(z2)  #m, 10

    # backpropagate to update weights
    d2 = np.multiply(np.subtract(a2, y), sigmoid(a2, deriv=True))  # m, 10
    d1 = np.multiply(d2.dot(theta2)[:,1:], sigmoid(z1, deriv=True))  # m, 25

    dtot1 = a0.T.dot(d1)  # 401, 25
    dtot2 = a1.T.dot(d2)  # 26, 10

    cost = (1 / m) * np.sum(np.multiply((-y), np.log(a2)) - np.multiply((1 - y), np.log(1 - a2))) + \
           (reg_lam / (2 * m)) * (np.sum(np.square(np.delete(theta1, 0, 1))) + 
                            np.sum(np.square(np.delete(theta2, 0, 1))))

    theta1_zeroed = np.concatenate((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]), axis=1)
    theta2_zeroed = np.concatenate((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]), axis=1)
    theta1_grad = ((1 / m) * dtot1).T + ((reg_lam / m) * theta1_zeroed)
    theta2_grad = ((1 / m) * dtot2).T + ((reg_lam / m) * theta2_zeroed)

    return cost, theta1_grad, theta2_grad


def main ():
    start_time = time.time()

    # load training data
    traind_file, trainl_file, testd_file, testl_file, theta_file = parse_args()
    X, y = read_train(traind_file, trainl_file)
    # initialize weights
    theta1, theta2 = load_theta(theta_file)

    epoch = 0
    error = 1

    while epoch < epochs and error > tolerance:
        # update weights
        error, t1grad, t2grad = update_weights(theta1, theta2, X, y, lam)
        theta1 += -alpha * t1grad
        theta2 += -alpha * t2grad

        # increment epoch and print error
        if epoch % 100 == 0:
            print("Epoch %d:  %s" % (epoch, str(error)))
        
        epoch += 1

    print("Accuracy: %s" % str(get_correct(theta1, theta2, X, y) / X.shape[0]))
    print('\nTime to train: ' + str(time.time() - start_time))

    demo_x = X[np.random.randint(0, X.shape[0]), :]
    print('\nThe predicted digit is: %s' % str(predict(theta1, theta2, demo_x)))
    show_image(demo_x)

    save_theta('/var/log/theta.mat', {'Theta1': theta1, 'Theta2': theta2})



'''
# main for opencl parallelization
def main ():
    # load data and initialize global variables
    traind_file, trainl_file, testd_file, testl_file = parse_args()
    train_data, train_labels = read_train(traind_file, trainl_file)
    a1 = np.zeros(input_size+1, np.float32)
    z2 = np.zeros(hidden_size, np.float32)
    a2 = np.zeros(hidden_size+1, np.float32)
    z3 = np.zeros(output_size, np.float32)
    a3 = np.zeros(output_size, np.float32)
    theta1 = np.random.rand(hidden_size, input_size+1)
    theta2 = np.random(output_size, hidden_size+1)
    
    # load opencl
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    # create the program to run with c++
    sigmoid = cl.Program(context, """
        __kernel void sigmoid (__global float *z, __global float *g) {
            int git = get_global_id(0);
            g[gid] = 1.0 / (1.0 + exp(z[gid]));
        }

        __kernel void matrix_dot_vector (__global float *theta, 
                                    __global float *X,
                                    __global float *activation) {
            int gid = get_global_id(0);
            activation[gid] = dot(theta[gid], X[0]);
        }

        /*__kernel void regularized_cost (__global float *theta1, 
                                        __global float *theta2,
                                        __global float *X
                                        __global float *y
                                        __global float *lambda,
                                        __global float *cost,
                                        __global float *grad) {

        }*/
        """).build()
    queue = cl.CommandQueue(context)

    # allocate device memory and move data from host to device
    mf = cl.mem_flags
    a1_buf = cl.Buffer(context, mf.READ_ONLY, a1.nbytes)
    z2_buf = cl.Buffer(context, mf.READ_WRITE, z2.nbytes)
    a2_buf = cl.Buffer(context, mf.READ_WRITE, a2.nbytes)
    z3_buf = cl.Buffer(context, mf.READ_WRITE, z3.nbytes)
    a3_buf = cl.Buffer(context, mf.READ_WRITE, a3.nbytes)
    theta1_buf = cl.Buffer(context, mf.READ_WRITE, theta1.nbytes)
    theta2_buf = cl.Buffer(context, mf.READ_WRITE, theta2.nbytes)
    # lam_buf = cl.Buffer(context, mf.READ_ONLY, lam_buf.nbytes)

    # train the model
    epoch = 0
    error = 1
    while epoch < epochs and error > tolerance:
        # feedforward

        # get cost, gradients and backpropogate to update weights

        epoch += 1
'''


if __name__ == '__main__':
    main()


