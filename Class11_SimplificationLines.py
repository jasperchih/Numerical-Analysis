#   # -*- coding: UTF-8 -*-
#   trial on the : Satomi machine
#   Created by Ush on 2018/5/18
#   Project name :  class10_ODE 
#   Please contact CHIH, HSIN-CHING/D0631008 when expect to refer this source code.
#   NOTE : no liability on any loss nor damage by using this source code. it is your own risk.

from __future__ import division
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import scipy.linalg as la
import numpy as np
import cmath
from rdp import rdp

# http://pycallgraph.readthedocs.io/en/master/examples/basic.html#source-code
from math import sqrt  # call sqrt from cmath for complex number
from numpy import matrix
from scipy.integrate import odeint
from pylab import *


class NCM11:
    def __init__(self, A, choice):
        "do something here"

    @staticmethod
    # https://zh.wikipedia.org/wiki/道格拉斯-普克算法
    # http://52north.github.io/wps-profileregistry/generic/dp-line-generalization.html
    # https://github.com/nramm/maskiton/blob/master/server/plugins/appion/pyami/douglaspeucker.py
    def RDP_middle(Px, Py, EPS):
        result_x = []
        result_y = []
        recResults1_X = []
        recResults1_Y = []
        recResults2_X = []
        recResults2_Y = []
        dmax,index = 0,0
        length = len(Py)
        for i in range(1, length - 2):
            d = NCM11.d(Px[0], Py[0], Px[i], Py[i], Px[length - 1], Py[length - 1])
            if (d > dmax):
                index = i
                dmax = d
        if (dmax >= EPS):
            # Recursive call
            recResults1_X, recResults1_Y = NCM11.RDP_middle(Px[: index + 1], Py[:index + 1], EPS)
            recResults2_X, recResults2_Y = NCM11.RDP_middle(Px[index:], Py[index:], EPS)
            # Build the result list
            result_x = np.vstack((recResults1_X[:-1], recResults2_X))
            result_y = np.vstack((recResults1_Y[:-1], recResults2_Y))
        else:
            result_x = np.vstack((Px[0], Px[-1]))
            result_y = np.vstack((Py[0], Py[-1]))
        return result_x, result_y

    @staticmethod
    # FMI : find middle index
    def FMI(Py):
        middle = float(len(Py)) / 2
        if middle % 2 != 0:
            middle = int(middle - 0.5)
        return middle

    @staticmethod
    # input : P  Polyline { P1, P2 ....Pn }, epsilon : offset
    # output :  list  simplification algorithms
    def rdp_Ramer_Douglas_Pecker(Px, Py, EPS):
        # https://pypi.org/project/rdp/
        # input : P  Polyline { P1, P2 ....Pn }, epsilon : offset
        # output :  list  simplification algorithms
        result = rdp(np.column_stack((Px, Py)), epsilon=EPS)
        return [row[0] for row in result], [row[1] for row in result]

    @staticmethod
    def Standard_Deviation_Method(Px, Py, EPS):
        result_x = []
        result_y = []
        MAF = []
        x_start = Px[0]
        y_start = Py[0]
        max_samples = 3
        EPS = EPS * 0.25
        result_x = np.append(result_x, x_start)
        result_y = np.append(result_y, y_start)
        p_size = Py.shape[0]
        for index in range(1, p_size - 1):
            Pack1x = np.array([Px[index - 1], Px[index], Px[index + 1]])
            SD1x = np.std(Pack1x)
            Pack1y = np.array([Py[index - 1], Py[index], Py[index + 1]])
            SD1y = np.std(Pack1y)
            MAF = np.append(MAF, sqrt(SD1x ** 2 + SD1y ** 2))
            Average = np.mean(MAF)
            if len(MAF) == max_samples:
                MAF = np.delete(MAF, 0)
            print(index, sqrt(SD1x ** 2 + SD1y ** 2), Average)
            if (sqrt(SD1x ** 2 + SD1y ** 2) - Average) > (EPS):
                result_x = np.append(result_x, Px[index])
                result_y = np.append(result_y, Py[index])
            else:
                pass
        result_x = np.append(result_x, Px[p_size - 1])
        result_y = np.append(result_y, Py[p_size - 1])
        return result_x, result_y

    @staticmethod
    def Simplification_Perpendicular_Distance(Px, Py, epsilon):
        # input : P  Polyline { P1, P2 ....Pn }, epsilon : offset
        # output :  list  simplification algorithms
        result_x = []
        result_y = []
        x_start = Px[0]
        y_start = Py[0]
        result_x = np.append(result_x, x_start)
        result_y = np.append(result_y, y_start)
        p_size = Py.shape[0]
        for index in range(1, p_size - 1):
            x_target = Px[index]
            y_target = Py[index]
            x_end = Px[index + 1]
            y_end = Py[index + 1]
            d_result = NCM11.d(x_start, y_start, x_target, y_target, x_end, y_end)
            if (d_result > epsilon):  # keep the original data and save into output vector
                result_x = np.append(result_x, Px[index])
                result_y = np.append(result_y, Py[index])
                x_start = Px[index]  # load the next number
                y_start = Py[index]
            else:  # skip the data
                pass
        # load the last data
        result_x = np.append(result_x, Px[p_size - 1])
        result_y = np.append(result_y, Py[p_size - 1])
        return result_x, result_y

    @staticmethod
    def d(x_s, y_s, x_target, y_target, x_end, y_end):
        norm_PsPe = sqrt((x_s - x_end ** 2 + (y_s - y_end) ** 2))
        norm_PePt = sqrt((x_end - x_target) ** 2 + (y_end - y_target) ** 2)
        norm_PtPs = sqrt((x_target - x_s) ** 2 + (y_target - y_s) ** 2)
        delta = abs(((x_end - x_s) * (y_target - y_s) - (x_target - x_s) * \
                     (y_end - x_s))) / (norm_PsPe)
        check = norm_PtPs ** 2 - delta ** 2 - norm_PsPe ** 2
        if (check <= 0):  # out of P2P1 line
            result = delta
        elif (norm_PtPs - norm_PePt) > 0:
            result = norm_PePt
        else:
            result = norm_PtPs
        return (result)


import matplotlib.pyplot as plt
import random


def main():
    print("let's make it happen")

    # Task 1 : the basic review
    if False:  # for the basic review

        X_data = []
        Y_data = []

        for index in range(0, 25):
            X_data = np.append(X_data, index)
            Y_data = np.append(Y_data, np.random.uniform(0., 15.))

        # start the task
        EPS = 2
        resultX1, resultY1 = NCM11.Simplification_Perpendicular_Distance(X_data, Y_data, EPS)
        resultX2, resultY2 = NCM11.rdp_Ramer_Douglas_Pecker(X_data, Y_data, EPS)
        resultX3, resultY3 = NCM11.Standard_Deviation_Method(X_data, Y_data, EPS)
        resultX4, resultY4 = NCM11.RDP_middle(X_data, Y_data, EPS)

        # plot with various axes scales
        plt.figure(figsize=(15, 9))

        # 221
        plt.subplot(221)
        plt.plot(X_data, Y_data, color='green', linestyle='dashed', linewidth=1.0, marker='d', markevery=1,
                 label='Original')
        plt.plot(resultX1, resultY1, color='red', linewidth=1.0, marker='o', markevery=1, label='Pre. method')
        plt.legend()
        plt.title(' Prependicular Method ')
        plt.gca().set_xticks(np.arange(0, 25, 1))
        plt.gca().set_yticks(np.arange(0, 16, 1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        # 222
        plt.subplot(222)
        plt.plot(X_data, Y_data, color='green', linestyle='dashed', linewidth=1.0, marker='d', markevery=1,
                 label='Original')
        plt.plot(resultX3, resultY3, color='red', linewidth=1.0, marker='o', markevery=1, label='STD method')
        plt.legend()
        plt.title(' Standard Deviation Method ')
        plt.gca().set_xticks(np.arange(0, 25, 1))
        plt.gca().set_yticks(np.arange(0, 16, 1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        # 223
        plt.subplot(223)
        plt.plot(X_data, Y_data, color='green', linestyle='dashed', linewidth=1.0, marker='d', markevery=1,
                 label='Original')
        plt.plot(resultX2, resultY2, color='red', linewidth=1.0, marker='o', markevery=1, label='RDP method')
        plt.legend()
        plt.title(' RDP-Dauglas-Peucker line Method (Python module) ')
        plt.gca().set_xticks(np.arange(0, 25, 1))
        plt.gca().set_yticks(np.arange(0, 16, 1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        # 224
        plt.subplot(224)
        plt.plot(X_data, Y_data, color='green', linestyle='dashed', linewidth=1.0, marker='d', markevery=1,
                 label='Original')
        plt.plot(resultX4, resultY4, color='red', linewidth=1.0, marker='o', markevery=1, label='RDP-Student method')

        plt.legend()

        plt.title(' OWN-Dauglas-Peucker line Method (Student module) ')
        plt.gca().set_xticks(np.arange(0, 25, 1))
        plt.gca().set_yticks(np.arange(0, 16, 1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        plt.show()

    # task 2 : loading Taiwan
    if True:

        l1_x = []
        l1_y = []

        file = open("taiwan.txt", "r")
        list = file.readlines()
        file.close()
        for index in list:
            tempX, tempY = int(index.split()[1]), 430 - int(index.split()[0])
            l1_x = np.append(l1_x, tempX)
            l1_y = np.append(l1_y, tempY)

        EPS = 3
        resultX1, resultY1 = NCM11.Simplification_Perpendicular_Distance(l1_x, l1_y, EPS * 0.4)
        resultX2, resultY2 = NCM11.rdp_Ramer_Douglas_Pecker(l1_x, l1_y, EPS * 0.8)
        resultX3, resultY3 = NCM11.Standard_Deviation_Method(l1_x, l1_y, EPS * 0.13)
        resultX4, resultY4 = NCM11.RDP_middle(l1_x, l1_y, EPS * 3.75)

        length0, length1, length2, length3, length4 = \
            len(l1_x), len(resultX1), len(resultX2), len(resultX3), len(resultX4)

        plt.plot(l1_x, l1_y, color='green', linewidth=1.0, marker='o', markevery=40, label='Original N=' + str(length0))
        plt.plot(resultX1, resultY1, color='red', linewidth=1.0, marker='+', markevery=20,
                 label='Pre. N=' + str(length1))
        plt.plot(resultX2, resultY2, color='blue', linewidth=1.0, marker='.', markevery=29,
                 label='RDP-Python. N=' + str(length2))
        plt.plot(resultX3, resultY3, color='pink', linewidth=2.0, marker='D', markevery=17,
                 label='STD. N=' + str(length3))
        plt.plot(resultX4, resultY4, color='black', linewidth=1.0, marker='*', markevery=33,
                 label='RDP-Student. N=' + str(length4))

        plt.legend()
        plt.title(' Taiwan Map in Hybrid method ')
        # plt.gca().set_xticks(np.arange(0, 25, 1))
        # plt.gca().set_yticks(np.arange(0, 16, 1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)


if __name__ == "__main__":
    main()
