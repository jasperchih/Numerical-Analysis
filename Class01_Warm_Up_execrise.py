#   # -*- coding: UTF-8 -*-
#   trial on the : \TBD
#   Created by Gakki on 2018/3/4
#   Project name :  class 1 
#   Please contact CHIH, HSIN-CHING/D0631008 when expect to refer this source code.
#   NOTE : no liability on any loss nor damage by using this source code. it is your own risk.
import numpy as np
from scipy.linalg import toeplitz


def matrix_diag():
    v = np.matrix(np.eye(16)) * -2
    w = np.matrix(np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1))
    u = np.matrix(np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], -1))
    diag_result = v + w + u
    diag_result[0, 15] = 1
    diag_result[15, 0] = 1
    print("Diag = \n" + str(diag_result))


def matrix_toeplitz():
    c = np.matrix([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    r = np.matrix([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    b_p = toeplitz(c, r) + np.transpose(toeplitz(c, r))
    print("Toeplitz = \n" + str(b_p))


import math

# definition :
# ax^2+bx+c=0
# x1/x2= (-b(+/-)sqrt(b^2-4ac))/2a
# https://stackoverflow.com/questions/10725522/arbitrary-precision-of-square-rootsfrom decimal import *
from bigfloat import *
import cmath


# error code 01 : invalid input as 0x^2+0x+c=0 format.
# error code 02 : invalid input as 0x^2+bx+c=0 format.
def quadratic_eq_root(a, b, c):
    # input data verification
    if (a != 0):
        eigen = (b ** 2 - 4 * a * c)
        if (eigen > 0):
            # real part only
            x1 = (-b + sqrt(eigen, precision(100))) / (2 * a)
            x2 = (-b - sqrt(eigen, precision(100))) / (2 * a)
        if (eigen == 0):
            # real part only
            x1, x2 = -b / (2 * a)
        if (eigen < 0):
            # real part and imaginary part
            x1 = (-b + cmath.sqrt(eigen)) / (2 * a)
            x2 = (-b - cmath.sqrt(eigen)) / (2 * a)
    else:
        if (b == 0):
            print("Error code 01  ")
            x1, x2 = 0, 0
        else:
            print("Error code 02  ")
            x1, x2 = 0, 0

    return (x1, x2)


# https://stackoverflow.com/questions/15390807/integer-square-root-in-python
def isqrt(num):
    x = num
    res = x
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + floor(num, x)) >> 1
        res = num - square(x)
    print(num, x, res)
    return x, res


# https://www.geeksforgeeks.org/calculate-square-of-a-number-without-using-and-pow/
def square(n):
    result = n
    if (n > 0):
        for i in range(1, n):
            result = result + n
    return result


# http://www.bogotobogo.com/python/python_interview_questions_2.php
def floor(a, b):
    count = 0
    sign = 1
    if a < 0: sign = -1
    while True:
        if b == 1: return a
        # positive
        if a >= 0:
            a = a - b
            if a < 0: break
        # negative
        else:
            a = -a - b
            a = -a
            if a > 0:
                count += 1
                break
        count += 1
    return count * sign


import time
import numpy
import matplotlib.pyplot as plt


def main():
    print
    "let's make it happen!"
    # Task1
    if True:
        avg1 = []
        avg2 = []
        for i in range(0, 1000):
            time_1 = time.time()
            matrix_diag()
            time_2 = time.time()
            matrix_toeplitz()
            time_3 = time.time()
            Diag_method_time = time_2 - time_1
            Toeplitz_method_time = time_3 - time_2
            avg1 = np.append(avg1, Diag_method_time)
            avg1_avg = np.mean(avg1)
            avg2 = np.append(avg2, Toeplitz_method_time)
            avg2_avg = np.mean(avg2)
            print(i)
        print("Diag_method: " + str(avg1_avg) + " second. Toeplitz method: " + str(avg2_avg) + " second")
    # Task2
    if True:
        x1, x2 = (quadratic_eq_root(1., -1.e+8, 1.))
        print("Task2 -own =")
        print(x1, x2)
        x1, x2 = (numpy.roots([1., -1.e+8, 1.]))
        print("Task2 -roots =")
        print(x1, x2)
    # Task3
    if True:
        root_own = []
        root_res = []
        root_libary = []
        fig, ax1 = plt.subplots()
        x = np.arange(0, 200, 1)
        for i in range(0, 200):
            root_own_data, root_own_res = isqrt(i)
            root_own = np.append(root_own, root_own_data)
            root_res = np.append(root_res, root_own_res)
            root_libary = np.append(root_libary, cmath.sqrt(i))
        ax1.plot(x, root_own, color='r', label='own code in root')
        ax1.plot(x, root_libary, color='g', label='cmath library')
        plt.legend(bbox_to_anchor=(0.35, 1))
        plt.ylim((0, 15))
        plt.ylabel('root value')
        plt.xlabel('input number')
        ax2 = ax1.twinx()
        ax2.bar(x, root_res, color='b', label='own code in residue')
        plt.xlabel('x')
        plt.ylim((0, 25))
        plt.ylabel('residure value')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0.8, 1))
        fig.tight_layout()
        plt.show()



if __name__ == "__main__":
    main()
