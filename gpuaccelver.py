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

'''


save_file = '/var/log/theta.mat'

image_width = 20
image_height = 20

input_size = 400
hidden_size = 25
output_size = 10

epochs = 10000
alpha = 0.3
tolerance = 0.30
lam = 1.0
epsilon_init = 0.12


def check_permissions ():
    if os.getuid():
        usage()


def read_file (filename, mode='r'):
    contents = []
    with open(filename, mode) as fname:
        for line in fname:
            contents.append(l.strip())

    return contents


def write_file (filename, data, mode='w'):
    with open(filename, mode) as fname:
        fname.writelines(data)


def show_image (x):
    reshaped = np.reshape(x, (20, 20))
    image = plt.imshow(reshaped, 'gray', origin='lower')
    plt.show()


def usage ():
    print('''
            sudo ./gpuaccelver.py --train training_data_file training_label_file
                                  [--test test_data_file test_label_file]
                                  [--theta theta_file] [-h, --help]
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


def sigmoid (z, deriv=False):
    if not deriv:
        return 1.0 / (1.0 + np.exp(-z))
    return sigmoid(z) * (1 - sigmoid(z))


def read_data (data_file, label_file, is_mat=True):
    if is_mat:
        data = spio.loadmat(data_file, squeeze_me=True)
        labels = spio.loadmat(label_file, squeeze_me=True)
        X = np.array([np.array(x).astype(np.float32) for x in data['X']]).astype(np.float32)
        # transform decimal label values to one-hot vectors
        tmp_y = np.array(labels['y'])
        y = np.zeros((tmp_y.shape[0], output_size), np.float32)
        for i, c in enumerate(tmp_y):
            y[i, c-1] = 1
        # or for demonstration
        # y = np.zeros((tmp_y.shape[0], output_size))
        # y[np.arange(tmp_y.shape[0]), tmp_y] = 1
    else:
        tmp_X = read_file(data_file)
        y = read_file(label_file)
        X = np.zeros((len(tmp_X), 400), np.float32)
        for i, x in enumerate(tmp_X):
            X[i] = tuple(x.split(','))

    return X, y


def load_theta (theta_file='', is_mat=True):
    if is_mat and theta_file:
        data = spio.loadmat(theta_file, squeeze_me=True)
        theta1 = np.asarray(data['Theta1'])
        theta2 = np.asarray(data['Theta2'])
    else:
        theta1 = np.random.rand(hidden_size, input_size + 1).astype(np.float32) \
                * 2 * epsilon_init - epsilon_init
        theta2 = np.random.rand(output_size, hidden_size + 1).astype(np.float32) \
                * 2 * epsilon_init - epsilon_init

    return theta1, theta2


def save_theta (theta_file, thetas):
    os.system('touch %s' % theta_file)
    spio.savemat(theta_file, thetas)


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


def main():
    # load training data
    traind_file, trainl_file, testd_file, testl_file, theta_file = parse_args()
    X, y = read_data(traind_file, trainl_file)
    # initialize weights
    theta1, theta2 = load_theta(theta_file)
    #initialize gradients
    theta1_grad = np.zeros(theta1.shape, np.float32)
    theta2_grad = np.zeros(theta2.shape, np.float32)

    epoch = 0
    error = np.ones(1, np.float32)

    # initialize opencl environment
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])

    # create the kernel to run
    program = cl.Program(context, '''

__kernel void sigmoid (__global float *z,
                       __global float *result) {
    int gid = get_global_id(0);
    result[gid] = 1.0 / (1.0 + exp(-z[gid]));
}

/*__kernel void update_weights (__global const float *theta1,
                                __global const float *theta2,
                                __global const float *X,
                                __global const float *y,
                                __global float *cost,
                                __global float *theta1_grad,
                                __global float *theta2_grad) {
    int gid = get_global_id(0);

    

}*/ 
        
        ''').build()

    queue = cl.CommandQueue(context)
    mf = cl.mem_flags

    # copy and convert data on host to cl-ready device
    X_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    test = np.zeros(X.shape, np.float32)
    cl_test = cl.Buffer(context, mf.WRITE_ONLY, test.nbytes)
    
    # cl_y = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    # cl_theta1 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta1)
    # cl_theta2 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta2)
    # cl_theta1_grad = cl.Buffer(context, mf.WRITE_ONLY, hostbuf=theta1_grad.nbytes)
    # cl_theta2_grad = cl.Buffer(context, mf.WRITE_ONLY, hostbuf=theta2_grad.nbytes)
    # cl_error = cl.Buffer(context, mf.WRITE_ONLY, hostbuf=error.nbytes)
    program.sigmoid(queue, test.shape, None, X_buf, cl_test)
    cl.enqueue_copy(queue, test, cl_test)
    print(sigmoid(X[0]))
    print(test[0])

    start_time = time.time()

    while epoch < epochs and error > tolerance:
        # run forward and back propagations on gpu to get error and weight gradients
        program.update_weights()
        # copy to host buffer
        cl.enqueue_copy(queue, error, error_buf)
        cl.enqueue_copy(queue, theta1_grad, theta1_grad_buf)
        cl.enqueue_copy(queue, theta2_grad, theta2_grad_buf)
        theta1 += -alpha * theta1_grad
        theta2 += -alpha * theta2_grad

        # increment epoch and print error
        if epoch % 100 == 0:
            print('Epoch %d:  %s' % (epoch, str(error)))
        epoch += 1

    print('\nTime to train: %s' % str(time.time() - start_time))
    print('\nAccuracy: %s' % str(get_correct(theta1, theta2, X, y) / X.shape[0]))

    demo_x = X[np.random.randint(0, X.shape[0]), :]
    print('\nThe predicted digit is: %s' % str(predict(theta1, theta2, demo_x)))
    show_image(demo_x)

    save_theta(save_file, {'Theta1': theta1, 'Theta2': theta2})


if __name__ == '__main__':
    main()

