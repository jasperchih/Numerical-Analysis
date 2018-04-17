#   # -*- coding: UTF-8 -*-
#   trial on the : Satomi machine
#   Created by Ush on 2018/3/23
#   Project name :  Class04 
#   Please contact CHIH, HSIN-CHING/D0631008 when expect to refer this source code.
#   NOTE : no liability on any loss nor damage by using this source code. it is your own risk.


import numpy as np
import math


def FUNC(type, x):
    F_select = 0
    formula = (0)
    if (type == 1):  # function1 is f(x)=x^2-1
        F_select = x ** 2 - 2  # 1,2
        formula = (1, 0, -2)
    elif (type == 2):  # 0,-2
        F_select = x ** 5 + x ** 3 + 3
        formula = (1, 0, 1, 0, 0, 3)
    elif (type == 3):  # 0,2
        F_select = x ** 3 - 2 * math.sin(x) - 1
    elif (type == 4):  # 1e-3, pi/2
        F_select = 0.5 * math.tan(x) - math.tanh(x)
    elif (type == 5):  # 3.1, 3.5
        F_select = (1 / (x - 3)) - 6
    elif (type == 6):
        F_select = x ** 3 + x ** 2 + x + 1
        formula = (1, 1, 1, 1)
    elif (type == 7):
        F_select = x ** 5 - 4.5 * x ** 4 + 4.55 * x ** 3 + 2.675 * x ** 2 - 3.3 * x - 1.4375
        formula = (1, -4.5, 4.55, 2.675, -3.3, -1.4375)
    else:
        print("invalid function")
    return (float(F_select))


import numpy as np
from numpy import *


# http://www2.gsu.edu/~matrhc/muller.py
def muller(f, p0, p1, p2, tol, max_iter=100):
    h1 = p1 - p0
    h2 = p2 - p1
    f_p1 = numpy.polyval(f, p1)
    f_p2 = numpy.polyval(f, p2)
    d1 = (f_p1 - numpy.polyval(f, p0)) / h1
    d2 = (f_p2 - f_p1) / h2
    d = (d2 - d1) / (h2 + h1)
    i = 2
    while i <= max_iter:
        b = d2 + h2 * d
        D = sqrt(b * b - 4 * f_p2 * d + 0j)
        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b - D
        h = -2 * f_p2 / E
        p = p2 + h
        if abs(h) < tol:
            return (p, i)
        p0 = p1
        p1 = p2
        p2 = p
        h1 = p1 - p0
        h2 = p2 - p1
        f_p1 = numpy.polyval(f, p1)
        f_p2 = numpy.polyval(f, p2)
        d1 = (f_p1 - numpy.polyval(f, p0)) / h1
        d2 = (f_p2 - f_p1) / h2
        d = (d2 - d1) / (h2 + h1)
        i += 1
    # print("Reached maximum number of iterations")
    return (p)


# http://math.oregonstate.edu/~restrepo/475A/Notes/sourcea-/node25.html
def Muller_solver(f, left, middle, right, tol, max_iter=50):
    iteration = 0
    while (abs(right - middle) > tol):
        if (iteration <= max_iter):
            P = np.polyfit([left, middle, right],
                           [numpy.polyval(f, left), numpy.polyval(f, middle), \
                            numpy.polyval(f, right)], 2)
            root = roots(P)
            left = middle
            middle = right
            if (abs(right - root[0]) < (abs(right - root[1]))):
                right = root[0]
            else:
                right = root[1]
                # print(iteration, left, middle, right)
        iteration = iteration + 1
    return (right, iteration)


from numpy.polynomial import polynomial as P
import numpy


# 分子：numerator 分母：denominator
def Muller_all_solver(f, left, middle, right, tol, max_iter=50):
    root_count = 0
    result = []
    rank = poly1d(f).order
    # print("# of solution should be :", rank)
    numerator = f
    while (root_count < rank):
        root1, itr = Muller_solver(numerator, left, middle, right, tol, 100)
        if isinstance(root1, complex):
            # print("root is complex")
            root1_conj = numpy.conjugate(root1)
            temp = (1, -1 * 2 * root1.real, +(root1 * root1_conj).real)
            result.append(root1)
            root_count = root_count + 1
            # print(root_count, root1)
            result.append(root1_conj)
            root_count = root_count + 1
            # print(root_count, root1_conj)
        else:
            # print("root is real")
            temp = (1, -1 * root1)
            result.append(root1)
            root_count = root_count + 1
            # print(root_count, root1)
        denominator = numpy.poly1d(temp)
        q, r = numpy.polydiv(numerator, denominator)
        numerator = q  # r value should be controlled in commerical case

    return (result)


from pprint import pprint


def main():
    # eps function in matlab
    # Python code expression : np.spacing(1)
    import warnings
    warnings.filterwarnings("ignore")
    formula2 = (1, 0, 1, 0, 0, 3)
    formula6 = (1, 1, 1, 1)
    formula7 = (1, -4.5, 4.55, 2.675, -3.3, -1.4375)
    print("value", numpy.polyval(formula2, 2))

    eps = np.spacing(1)
    eps = 1e-15
    print("Python eps : " + str(eps))

    print
    "let's make it happen!"
    print("formula (a) :")
    # print(numpy.poly1d(formula2))
    print("Result1:  \t\t\t\t: ", Muller_solver(formula2, 0, 1.5, 2, eps, 100))
    print("Result2:  \t\t\t\t: ", muller(formula2, 0, 1.5, 2, eps, 100))
    print("formula (b) :")
    # print(numpy.poly1d(formula6))
    print("Result1:  \t\t\t\t: ", Muller_solver(formula6, 0, 1.5, 2, eps, 100))
    print("Result2:  \t\t\t\t: ", muller(formula6, 0, 1.5, 2, eps, 100))
    print("formula (c) :")
    # print(numpy.poly1d(formula7))
    print("Result1:  \t\t\t\t: ", Muller_solver(formula7, 0, 1.5, 2, eps, 100))
    print("Result2:  \t\t\t\t: ", muller(formula7, 0, 1.5, 2, eps, 100))

    print(numpy.poly1d(formula2))
    print("Muller method in all roots \t:", Muller_all_solver(formula2, 0, 1.5, 2, eps, 100))
    print(numpy.poly1d(formula6))
    print("Muller method in all roots \t:", Muller_all_solver(formula6, 0, 1.5, 2, eps, 100))
    print(numpy.poly1d(formula7))
    print("Muller method in all roots \t:", Muller_all_solver(formula7, 0, 1.5, 2, eps, 100))



if __name__ == "__main__":
    main()
