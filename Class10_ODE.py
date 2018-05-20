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

# http://pycallgraph.readthedocs.io/en/master/examples/basic.html#source-code
from math import sqrt  # call sqrt from cmath for complex number
from numpy import matrix
from scipy.integrate import odeint
from pylab import *


class NCM10:
    def __init__(self, A, choice):
        "do something here"

    # https://blog.csdn.net/caimouse/article/details/78043518
    # http://people.bu.edu/andasari/courses/numericalpython/python.html
    @staticmethod
    def rk3_solver(function, t0, tf, yinit):
        t = t0
        end = tf
        h = 0.01
        y_now = yinit
        time = []
        y_dot = []
        y = []
        t = t + h
        while (t < end):
            k1 = function(y_now, t) * h
            k2 = function(y_now + 0.5 * k1, t + 0.5 * h) * h
            k3 = function(y_now - k1 + 2 * k2, t + h) * h
            y_now = y_now + ((k1 + 4 * k2 + k3) / 6.)
            time = np.append(time, t)
            y_dot = np.append(y_dot, k1)
            y = np.append(y, y_now)
            t = t + h
        return time, y, y_dot

    @staticmethod
    def rk4_solver(function, t0, tf, yinit):
        t = t0
        end = tf
        h = 0.01
        y_now = yinit
        time = []
        y_dot = []
        y = []
        t = t + h
        while (t < end):
            k1 = function(y_now, t)
            k2 = function(y_now + 0.5 * k1 * h, t + 0.5 * h)
            k3 = function(y_now + 0.5 * k2 * h, t + 0.5 * h)
            k4 = function(y_now + 1.0 * k3 * h, t + 1.0 * h)
            y_now = y_now + ((k1 + 2 * k2 + 2 * k3 + k4) / 6.) * h
            time = np.append(time, t)
            y_dot = np.append(y_dot, k1)
            y = np.append(y, y_now)
            t = t + h
        return time, y, y_dot

    @staticmethod
    def multi3_solver(function, t0, tf, yinit):
        t = t0
        end = tf
        h = 0.01
        y_now = yinit
        time = []
        y_dot = []
        y = []
        t = t + h
        while (t < end):
            m1 = (23. / 12) * function(y_now, t)
            m2 = (-16. / 12 * function(y_now, t - h))
            m3 = (5. / 12) * function(y_now, t - 2 * h)
            y_now = y_now + (m1 + m2 + m3) * h
            time = np.append(time, t)
            y_dot = np.append(y_dot, function(y_now, t))
            y = np.append(y, y_now)
            t = t + h
        return time, y, y_dot

    @staticmethod
    def Euler_solver(function, t0, tf, yinit):
        t = t0
        end = tf
        h = 0.01
        y_now = yinit # for only 1 order ODE.
        time = []
        y_dot = []
        y = []
        t = t + h
        while (t < end):
            Euler_step = function(y_now, t)
            y_now = y_now + np.dot(Euler_step, h)
            time = np.append(time, t)
            y_dot = np.append(y_dot, Euler_step)
            y = np.append(y, y_now)
            t = t + h
        return time, y, y_dot

    @staticmethod
    def rk1_solver_core(function, t0, tf, yinit):
        t = t0
        h = 1. / 1000

    @staticmethod
    # function that returns dy/dt
    # dy(t)/dt=-0.3*y(y)
    # https://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations
    def deriv1(y, t):
        dydt = -0.3 * y
        return dydt

    @staticmethod
    # function that returns dy/dt
    # y"-2*y'-0.1*y=0
    # [x',y'].T=[[2,1],[1,0]]*[x,y].T
    def deriv2(y, x):  # 返回值是y和y的导数组成的数组
        dydx = y * sin(2.5 * x) + x * cos(2.5 * x)
        return (dydx)

    @staticmethod
    # http://people.revoledu.com/kardi/tutorial/ODE/Runge%20Kutta%203.htm
    def deriv3(y, x):  # 返回值是y和y的导数组成的数组
        dydx = -((2 / x) + y ** 2) / (2 * x * y)
        return (dydx)

    @staticmethod
    # function that returns dy/dt
    def deriv4(y, x):  # 返回值是y和y的导数组成的数组
        dydx = -((2 / x) + y ** 2) * sin(x)
        return dydx


import matplotlib.pyplot as plt


def main():
    print("let's make it happen")

    time, y, y_dot = NCM10.Euler_solver(NCM10.deriv1, 1, 20, 1)
    Euler_result1 = y
    time, y, y_dot = NCM10.rk3_solver(NCM10.deriv1, 1, 20, 1)
    RK3_result1 = y
    time, y, y_dot = NCM10.rk4_solver(NCM10.deriv1, 1, 20, 1)
    RK4_result1 = y
    time, y, y_dot = NCM10.multi3_solver(NCM10.deriv1, 1, 20, 1)
    multi3_result1 = y
    time1 = time

    time, y, y_dot = NCM10.Euler_solver(NCM10.deriv2, 1, 50, -10)
    Euler_result2 = y
    time, y, y_dot = NCM10.rk3_solver(NCM10.deriv2, 1, 50, -10)
    RK3_result2 = y
    time, y, y_dot = NCM10.rk4_solver(NCM10.deriv2, 1, 50, -10)
    RK4_result2 = y
    time, y, y_dot = NCM10.multi3_solver(NCM10.deriv2, 1, 50, -10)
    multi3_result2 = y
    time2 = time

    time, y, y_dot = NCM10.Euler_solver(NCM10.deriv3, 0.1, 10, 1)
    Euler_result3 = y
    time, y, y_dot = NCM10.rk3_solver(NCM10.deriv3, 0.1, 10, 1)
    RK3_result3 = y
    time, y, y_dot = NCM10.rk4_solver(NCM10.deriv3, 0.1, 10, 1)
    RK4_result3 = y
    time, y, y_dot = NCM10.multi3_solver(NCM10.deriv3, 0.1, 10, 1)
    multi3_result3 = y
    time3 = time

    time, y, y_dot = NCM10.Euler_solver(NCM10.deriv4, 1, 25, 1)
    Euler_result4 = y
    time, y, y_dot = NCM10.rk3_solver(NCM10.deriv4, 1, 25, 1)
    RK3_result4 = y
    time, y, y_dot = NCM10.rk4_solver(NCM10.deriv4, 1, 25, 1)
    RK4_result4 = y
    time, y, y_dot = NCM10.multi3_solver(NCM10.deriv4, 1, 25, 1)
    multi3_result4 = y
    time4 = time

    # plot with various axes scales
    plt.figure(figsize=(10, 8))

    # linear
    plt.subplot(221)
    plt.plot(time1, Euler_result1, color='red', linewidth=1.0, marker='.', markevery=800, label='Euler')
    plt.plot(time1, RK3_result1, color='green', linewidth=1.0, marker='o', markevery=1000, label='RK3')
    plt.plot(time1, RK4_result1, color='blue', linewidth=1.0, marker='+', markevery=600, label='RK4')
    plt.plot(time1, multi3_result1, color='pink', linewidth=1.0, marker='d', markevery=750, label='Mul3')
    plt.legend()

    plt.title('Equation-1  dy(t)/dt=-0.3*y(y)')
    plt.grid(True)

    # log
    plt.subplot(222)
    plt.plot(time2, Euler_result2, color='red', linewidth=1.0, marker='.', markevery=800, label='Euler')
    plt.plot(time2, RK3_result2, color='green', linewidth=1.0, marker='o', markevery=1000, label='RK3')
    plt.plot(time2, RK4_result2, color='blue', linewidth=1.0, marker='+', markevery=600, label='RK4')
    plt.plot(time2, multi3_result2, color='pink', linewidth=1.0, marker='d', markevery=750, label='Mul3')
    plt.legend()

    plt.title('Equation-2  dydx = y * sin(2.5 * x) + x * cos(2.5 * x)')
    plt.grid(True)

    # symmetric log
    plt.subplot(223)
    plt.plot(time3, Euler_result3, color='red', linewidth=1.0, marker='.', markevery=180, label='Euler')
    plt.plot(time3, RK3_result3, color='green', linewidth=1.0, marker='o', markevery=200, label='RK3')
    plt.plot(time3, RK4_result3, color='blue', linewidth=1.0, marker='+', markevery=160, label='RK4')
    plt.plot(time3, multi3_result3, color='pink', linewidth=1.0, marker='d', markevery=150, label='Mul3')
    plt.legend()

    plt.title('Equation-3  dydx = -((2/x) +y^2) / (2*x*y)')
    plt.grid(True)

    # logit
    plt.subplot(224)
    plt.plot(time4, Euler_result4, color='red', linewidth=1.0, marker='.', markevery=1000, label='Euler')
    plt.plot(time4, RK3_result4, color='green', linewidth=1.0, marker='o', markevery=800, label='RK3')
    plt.plot(time4, RK4_result4, color='blue', linewidth=1.0, marker='+', markevery=600, label='RK4')
    plt.plot(time4, multi3_result4, color='pink', linewidth=1.0, marker='d', markevery=700, label='Mul3')
    plt.legend()

    plt.title('Equation-4  dydx = -((2 / x) + y ** 2) *sin(x)')
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
