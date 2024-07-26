import numpy as np
import time

size = 300
A = np.random.rand(size, size)
B = np.random.rand(size, size)

start_time = time.time()
C_dot = np.dot(A, B)
dot_time = time.time() - start_time

C_loop = np.zeros((size, size))
start_time = time.time()
for i in range(size):
    for j in range(size):
        for k in range(size):
            C_loop[i, j] += A[i, k] * B[k, j]
loop_time = time.time() - start_time

print(dot_time, loop_time)
