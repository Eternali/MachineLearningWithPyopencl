#!/usr/bin/python3

import numpy as np
import pyopencl as cl
from pyopencl import array


# initialize opencl environment
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])

# create the kernel to run
program = cl.Program(context, '''
    __kernel void sigmoid(__global const float4 *z,
    __global float *result)
    {
        int gid = get_global_id(0);
        result = z;
    }
    ''').build()

queue = cl.CommandQueue(context)
mf = cl.mem_flags

# copy and convert data on host to cl-ready device
vector = np.zeros((1, 1), cl.array.vec.float4)
vector[0, 0] = (1, 2, 4, 8)
matrix_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vector)
test = np.zeros(4, np.float32)
cl_test = cl.Buffer(context, mf.WRITE_ONLY, test.nbytes)

program.sigmoid(queue, test.shape, None, matrix_buf, cl_test)
cl.enqueue_copy(queue, test, cl_test)
print(test)
