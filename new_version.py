#!/usr/bi.python3

import matplotlib as mpl
import numpy as np
import os
import pyopencl as cl
import scipy.io as spio
import sys
import time

'''

'''

##----  CONSTANTS  ----##



##----  HELPERS  ----##



##----  KERNEL  ----##

kernel = '''
// dot product
__kernel void dot(
        const unsigned int size,
        __global float* A,
        __global float* B,
        __global float* C) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    C[i + size * j] = 0;

    for (int k = 0; k < size; k++) {
        C[i + size * j] += A[k + size * i] * B[j + size * k];
    }
}

// transpose
__kernel void transpose(
        __global float* A_T,
        __global float* A,
        const unsigned int width,
        const unsigned int height) {
    int read_idx = get_global_id(0) + get_global_id(1) * width;
    int write_idx = get_global_id(1) + get_global_id(0) * height;

    A_T[write_idx] = A[read_idx];
    }

// matrix multiplication
__kernel void mmul(
        const int N,
        __global float* A,
        __global float* B,
        __global float* C) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0;
    if (i < N && j < N) {
        tmp = 0.0f;
        for (int k = 0; k < N; k++) {
            tmp += A[i*N + k] * B[k*N + j];
        }
        C[i*N + j] = tmp;
    }
}

// element-wise sigmoid operation
__kernel void sigmoid(
        __global float* A,
        __global float* B) {
    int i = get_global_id(0);
    B[i] = 1.0f / (1.0f + exp(-A[i]));
}
'''


##----  MAIN  ----##

def main():
    # initialize opencl environment
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])

    # load the kernel
    program = cl.Program(context, kernel)

    # initialize runtime variables
    queue = cl.CommandQueue(context)
    mf = cl.mem_flags

    # load commandline arguments
    train_file, test_file, theta_file = parse_args()

    # load training data
    X, y = read_data(train_file)
    Xt, yt = read_data(test_file)
    m = X.shape[0]

    # load network metadata
    MAX_EPOCS = 1000
    TOLERANCE = np.array([0.3], np.float32)
    alpha = 0.1
    epoch = 0
    error = np.ones(1, np.float32)

    # initialize network weights
    theta1, theta2 = load_theta(theta_file)
    theta1T = np.array(theta1.T, np.float32)
    theta2T = np.array(theta2.T, np.float32)
    theta1_grad = np.zeros(theta1.shape, np.float32)
    theta2_grad = np.zeros(theta2.shape, np.float32)

    # initialize network layers
    a0 = np.zeros((m, X.shape[1]+1), np.float32)            # m, 401
    z1 = np.zeros((m, theta1.shape[0]), np.float32)         # m, 25
    a1 = np.zeros((m, theta1.shape[0] + 1), np.float32)     # m, 26
    z2 = np.zeros((m, theta1.shape[0]), np.float32)         # m, 10
    a2 = np.zeros((m, theta2.shape[0] + 1), np.float32)     # m, 10

    #initialize opencl buffers
    Xb = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hustbuf=X)
    yb = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    theta1b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta1)
    theta2b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta2)
    theta1Tb = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta1T)
    theta2Tb = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta2T)
    a0b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a0)
    z1b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z1)
    a1b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a1)
    z2b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z2)
    a2b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2)

    # start training timer
    start_time = time.time()

    while epoch < MAX_EPOCS and error > TOLERANCE:

        # forward propogation to calculate activations
        a0 = np.insert(X, 0, 1, axis=1)
        program.dot(queue, a0.shape, None, np.int32(len(a0)), a0b, theta1Tb, z1b)
        cl.enqueue_copy(queue, z1, z1b)
        program.sigmoid(queue, z1.shape, None, z1b, a1b)
        cl.enqueue_copy(queue, a1, a1b)

        a1 = np.insert(a1, 0, 1, axis=1)
        program.dot(queue, a1.shape, None, np.int32(len(a1)), a1b, theta2Tb z2b)
        cl.enqueue_copy(queue, z2, z2b)
        program.sigmoid(queue, z2.shape, None, z2b, a2b)
        cl.enqueue_copy(queue, a2, a2b)

        # error calculation


        # update_weights
        theta1 -= alpha * theta1_grad
        theta2 -= alpha * theta2_grad

        # increment epoch and show progress
        if epoch % 100 == 0:
            print('Epoch {}: {}'.format(epoch, error))
        epoch += 1

    # save trained network data
    save_theta(save_file, { 'Theta1': theta1, 'Theta2': theta2 })
    save_log(time.time() - start_time, alpha, epoch, error)

    # show stats on trained network
    print('')
    print('Time to train: {}'.format(time.time() - start_time))
    print('Accuracy: {}'.format(predict(X, y, [theta1, theta2])['accuracy']))
    print('')

    # demo single datapoint
    demo_idx = np.random.randint(0, X.shape[0])
    demo_x = X[demo_idx, :]
    print('Predicted digit is: {}'.format(predict(demo_x, y[demo_idx], [theta1, theta2])))
    show_image(demo_x)


if __name__ == '__main__':
    main()
