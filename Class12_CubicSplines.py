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
from scipy.interpolate import CubicSpline


# http://mropengate.blogspot.com/2015/04/cubic-spline-interpolation.html
class NCM12:
    def __init__(self, A, choice):
        "do something here"

    @staticmethod
    def Problem1(x, y):
        # input SegmentVectors, and x,y data
        # output the line points inside the Segments
        resolution = np.arange(0, 3.1, 0.1)
        X = np.linspace(x[0], x[1] - 0.1, 10)
        h = x[1] - x[0]
        ss0 = (y[0] * (x[1] - X) / h) + y[1] * (X - x[0]) / h

        X = np.linspace(x[1], x[2] - 0.1, 10)
        h = x[2] - x[1]
        ss1 = (y[1] * (x[2] - X) / h) + y[2] * (X - x[1]) / h

        X = np.linspace(x[2], x[3], 11)
        h = x[3] - x[2]
        ss2 = (y[2] * (x[3] - X) / h) + y[3] * (X - x[2]) / h

        ss0 = np.append(ss0, ss1)
        ss0 = np.append(ss0, ss2)

        return x, y, resolution, ss0

    @staticmethod
    def LineDrawing(s0, s1, s2, x, y):
        # input SegmentVectors, and x,y data
        # output the line points inside the Segments

        X = np.linspace(x[0], x[1] - 0.1, 10)
        ss0 = s0[0] * (X - x[0]) ** 3 + s0[1] * (X - x[0]) ** 2 + s0[2] * (X - x[0]) + s0[3]
        X = np.linspace(x[1], x[2] - 0.1, 10)
        ss1 = s1[0] * (X - x[1]) ** 3 + s1[1] * (X - x[1]) ** 2 + s1[2] * (X - x[1]) + s1[3]
        X = np.linspace(x[2], x[3], 11)
        ss2 = s2[0] * (X - x[2]) ** 3 + s2[1] * (X - x[2]) ** 2 + s2[2] * (X - x[2]) + s2[3]
        ss0 = np.append(ss0, ss1)
        ss0 = np.append(ss0, ss2)
        return ss0

    @staticmethod
    # 首尾兩端點的微分值是被指定的，這裡分別定為A和B。
    def Problem2_curve1_clamped_cubic(x, y, t_initial, t_end):
        # input x, y
        # output x,  curve CS in cs, cs^(1), cs^(2)
        # The first derivative at curves ends are zero
        cs = CubicSpline(x, y, bc_type=((1, t_initial), (1, t_end)))
        resolution = np.arange(0, 3.1, 0.1)
        print(len(resolution))
        # start the own code
        d = []
        u = [0]
        h = [1, 1, 1]  # this should be use the x[k+1]-x[k] in the real application
        size = len(x)
        for index in range(0, size - 1):
            d = np.append(d, (y[index + 1] - y[index]) / h[index])
        for index in range(0, size - 2):
            u = np.append(u, 6 * (d[index + 1] - d[index]))
        index, jndex = 0, 1
        b = np.matrix([
            [u[index + 1] - 3 * (d[index] - t_initial)],
            [u[size - 2] - 3 * (t_end - d[size - 2])]
        ])
        A = np.matrix([
            [(3 / 2) * h[0] + 2 * h[1], h[1]],
            [h[1], 2 * h[1] + (3 / 2) * h[1]]
        ])
        xx = NCM07.choleski(A, b)
        # table 5-8 clamped spline
        m0 = ((3 / h[0]) * (d[0] - t_initial)) - (1 / 2) * xx[0]
        m3 = (((3 / h[1]) * (t_end - d[2])) - ((1 / 2) * xx[1]))
        # packing the m-list
        m = [m0.item(0), xx.item(0), xx.item(1), m3.item(0)]
        # original function   -- create the function in piece-wise s0, s1 and s2
        s0 = [(m[1] - m[0]) / (6 * h[0]), m[0] / 2, ((y[1] - y[0]) / h[0]) - h[0] * (2 * m[0] + m[1]) / 6, y[0]]
        s1 = [(m[2] - m[1]) / (6 * h[1]), m[1] / 2, ((y[2] - y[1]) / h[1]) - h[1] * (2 * m[1] + m[2]) / 6, y[1]]
        s2 = [(m[3] - m[2]) / (6 * h[2]), m[2] / 2, ((y[3] - y[2]) / h[2]) - h[2] * (2 * m[2] + m[3]) / 6, y[2]]
        f0 = NCM12.LineDrawing(s0, s1, s2, x, y)
        # first order derivate
        s0_1df = [0, s0[0] * 3, s0[1] * 2, s0[2]]
        s1_1df = [0, s1[0] * 3, s1[1] * 2, s1[2]]
        s2_1df = [0, s2[0] * 3, s2[1] * 2, s2[2]]
        f1 = NCM12.LineDrawing(s0_1df, s1_1df, s2_1df, x, y)
        # second order derivate
        s0_2df = [0, 0, s0_1df[1] * 2, s0_1df[2]]
        s1_2df = [0, 0, s1_1df[1] * 2, s1_1df[2]]
        s2_2df = [0, 0, s2_1df[1] * 2, s2_1df[2]]
        # print("s0_2df, s1_2df, s2_2df :", s0_2df, s1_2df, s2_2df)
        f2 = NCM12.LineDrawing(s0_2df, s1_2df, s2_2df, x, y)

        return resolution, cs(resolution), cs(resolution, 1), cs(resolution, 2), f0, f1, f2

    @staticmethod
    # 首尾兩端沒有受到任何讓它們彎曲的力。
    def Problem2_curve2_natural(x, y, t_initial, t_end):
        cs = CubicSpline(x, y, bc_type=((2, t_end), (2, t_initial)))
        resolution = np.arange(0, 3.1, 0.1)
        # start the own code
        d = []
        u = [0]
        h = [1, 1, 1]  # this should be use the x[k+1]-x[k] in the real application
        size = len(x)
        for index in range(0, size - 1):
            d = np.append(d, (y[index + 1] - y[index]) / h[index])
        for index in range(0, size - 2):
            u = np.append(u, 6 * (d[index + 1] - d[index]))
        b = np.matrix([
            [u.item(1)],
            [u.item(2)]
        ])
        # Lemma 5-2
        A = np.matrix([
            [2 * (h[0] + h[1]), h[1]],
            [h[size - 3], 2 * (h[size - 3] + h[size - 2])]
        ])
        xx = NCM07.choleski(A, b)
        # table 5-8 Natual spline
        m0 = 0
        m3 = 0
        # packing the m-list
        m = [m0, xx.item(0), xx.item(1), m3]
        # original function ,create the function in piece-wise s0, s1 and s2
        s0 = [(m[1] - m[0]) / (6 * h[0]), m[0] / 2, ((y[1] - y[0]) / h[0]) - h[0] * (2 * m[0] + m[1]) / 6, y[0]]
        s1 = [(m[2] - m[1]) / (6 * h[1]), m[1] / 2, ((y[2] - y[1]) / h[1]) - h[1] * (2 * m[1] + m[2]) / 6, y[1]]
        s2 = [(m[3] - m[2]) / (6 * h[2]), m[2] / 2, ((y[3] - y[2]) / h[2]) - h[2] * (2 * m[2] + m[3]) / 6, y[2]]
        f0 = NCM12.LineDrawing(s0, s1, s2, x, y)
        # first order derivate
        s0_1df = [0, s0[0] * 3, s0[1] * 2, s0[2]]
        s1_1df = [0, s1[0] * 3, s1[1] * 2, s1[2]]
        s2_1df = [0, s2[0] * 3, s2[1] * 2, s2[2]]
        f1 = NCM12.LineDrawing(s0_1df, s1_1df, s2_1df, x, y)
        # second order derivate
        s0_2df = [0, 0, s0_1df[1] * 2, s0_1df[2]]
        s1_2df = [0, 0, s1_1df[1] * 2, s1_1df[2]]
        s2_2df = [0, 0, s2_1df[1] * 2, s2_1df[2]]
        f2 = NCM12.LineDrawing(s0_2df, s1_2df, s2_2df, x, y)
        return resolution, cs(resolution), cs(resolution, 1), cs(resolution, 2), f0, f1, f2

    @staticmethod
    def Problem2_curve2_NotAKnot(x, y):
        cs = CubicSpline(x, y)
        resolution = np.arange(0, 3.1, 0.1)
        # start the own code
        d = []
        u = [0]
        h = [1, 1, 1]  # this should be use the x[k+1]-x[k] in the real application
        size = len(x)
        for index in range(0, size - 1):
            d = np.append(d, (y[index + 1] - y[index]) / h[index])
        for index in range(0, size - 2):
            u = np.append(u, 6 * (d[index + 1] - d[index]))
        b = np.matrix([
            [u.item(1)],
            [u.item(2)]
        ])
        # Lemma 5-3
        A = np.matrix([
            [3 * h[0] + 2 * h[1] + h[0] ** 2 / h[1], h[1] - h[0] ** 2 / h[1]],
            [h[size - 3] - h[size - 2] ** 2 / h[size - 3],
             2 * h[size - 3] + 3 * h[size - 2] + (h[size - 2] ** 2 / h[size - 3])]
        ])
        xx = NCM07.choleski(A, b)
        # table 5-8 (iii) Not-A-Knot spline
        m0 = xx.item(0) - (h[0] / h[1]) * (xx.item(1) - xx.item(0))
        m3 = xx.item(1) + (h[2] / h[1]) * (-xx.item(0) + xx.item(1))
        # packing the m-list
        m = [m0, xx.item(0), xx.item(1), m3]
        # original function
        s0 = [(m[1] - m[0]) / (6 * h[0]), m[0] / 2, ((y[1] - y[0]) / h[0]) - h[0] * (2 * m[0] + m[1]) / 6, y[0]]
        s1 = [(m[2] - m[1]) / (6 * h[1]), m[1] / 2, ((y[2] - y[1]) / h[1]) - h[1] * (2 * m[1] + m[2]) / 6, y[1]]
        s2 = [(m[3] - m[2]) / (6 * h[2]), m[2] / 2, ((y[3] - y[2]) / h[2]) - h[2] * (2 * m[2] + m[3]) / 6, y[2]]
        f0 = NCM12.LineDrawing(s0, s1, s2, x, y)
        # first order derivate
        s0_1df = [0, s0[0] * 3, s0[1] * 2, s0[2]]
        s1_1df = [0, s1[0] * 3, s1[1] * 2, s1[2]]
        s2_1df = [0, s2[0] * 3, s2[1] * 2, s2[2]]
        f1 = NCM12.LineDrawing(s0_1df, s1_1df, s2_1df, x, y)
        # second order derivate
        s0_2df = [0, 0, s0_1df[1] * 2, s0_1df[2]]
        s1_2df = [0, 0, s1_1df[1] * 2, s1_1df[2]]
        s2_2df = [0, 0, s2_1df[1] * 2, s2_1df[2]]
        f2 = NCM12.LineDrawing(s0_2df, s1_2df, s2_2df, x, y)
        return resolution, cs(resolution), cs(resolution, 1), cs(resolution, 2), f0, f1, f2

    @staticmethod
    def Problem2_curve2_ParabolicallyTerminate(x, y):
        cs = CubicSpline(x, y)
        resolution = np.arange(0, 3.1, 0.1)
        # start the own code
        d = []
        u = [0]
        h = [1, 1, 1]  # this should be use the x[k+1]-x[k] in the real application
        size = len(x)
        print("size : ", size)
        for index in range(0, size - 1):
            d = np.append(d, (y[index + 1] - y[index]) / h[index])
        for index in range(0, size - 2):
            u = np.append(u, 6 * (d[index + 1] - d[index]))
        b = np.matrix([
            [u.item(1)],
            [u.item(2)]
        ])
        # Lemma 5-4
        A = np.matrix([
            [3 * h[0] + 2 * h[1], h[1]],
            [h[size - 3], 2 * h[size - 3] + 3 * h[size - 2]]
        ])
        xx = NCM07.choleski(A, b)
        m0 = xx.item(0)
        m3 = xx.item(1)
        # packing the m-list
        m = [m0, xx.item(0), xx.item(1), m3]
        # original function
        s0 = [(m[1] - m[0]) / (6 * h[0]), m[0] / 2, ((y[1] - y[0]) / h[0]) - h[0] * (2 * m[0] + m[1]) / 6, y[0]]
        s1 = [(m[2] - m[1]) / (6 * h[1]), m[1] / 2, ((y[2] - y[1]) / h[1]) - h[1] * (2 * m[1] + m[2]) / 6, y[1]]
        s2 = [(m[3] - m[2]) / (6 * h[2]), m[2] / 2, ((y[3] - y[2]) / h[2]) - h[2] * (2 * m[2] + m[3]) / 6, y[2]]
        f0 = NCM12.LineDrawing(s0, s1, s2, x, y)
        # first order derivate
        s0_1df = [0, s0[0] * 3, s0[1] * 2, s0[2]]
        s1_1df = [0, s1[0] * 3, s1[1] * 2, s1[2]]
        s2_1df = [0, s2[0] * 3, s2[1] * 2, s2[2]]
        f1 = NCM12.LineDrawing(s0_1df, s1_1df, s2_1df, x, y)
        # second order derivate
        s0_2df = [0, 0, s0_1df[1] * 2, s0_1df[2]]
        s1_2df = [0, 0, s1_1df[1] * 2, s1_1df[2]]
        s2_2df = [0, 0, s2_1df[1] * 2, s2_1df[2]]
        f2 = NCM12.LineDrawing(s0_2df, s1_2df, s2_2df, x, y)
        return resolution, f0, f1, f2

    @staticmethod
    def Problem2_curve2_EndpointCurvatureAdjustSpline(x, y, t_initial, t_end):
        # cs = CubicSpline(x, y)
        resolution = np.arange(0, 3.1, 0.1)
        # start the own code
        d = []
        u = [0]
        h = [1, 1, 1]  # this should be use the x[k+1]-x[k] in the real application
        size = len(x)
        for index in range(0, size - 1):
            d = np.append(d, (y[index + 1] - y[index]) / h[index])
        for index in range(0, size - 2):
            u = np.append(u, 6 * (d[index + 1] - d[index]))
        # Lemma 5-5
        b = np.matrix([
            [u.item(1) - h[0] * t_initial],
            [u.item(2) - h[size - 2] * t_end]
        ])
        A = np.matrix([
            [2 * (h[0] + h[1]), h[1]],
            [h[size - 3], 2 * (h[size - 3] + h[size - 2])]
        ])
        xx = NCM07.choleski(A, b)
        # table 5-8  (v) Endpoint curvature adjust Spline
        m0 = t_initial
        m3 = t_end
        # packing the m-list
        m = [m0, xx.item(0), xx.item(1), m3]
        # original function
        s0 = [(m[1] - m[0]) / (6 * h[0]), m[0] / 2, ((y[1] - y[0]) / h[0]) - h[0] * (2 * m[0] + m[1]) / 6, y[0]]
        s1 = [(m[2] - m[1]) / (6 * h[1]), m[1] / 2, ((y[2] - y[1]) / h[1]) - h[1] * (2 * m[1] + m[2]) / 6, y[1]]
        s2 = [(m[3] - m[2]) / (6 * h[2]), m[2] / 2, ((y[3] - y[2]) / h[2]) - h[2] * (2 * m[2] + m[3]) / 6, y[2]]
        f0 = NCM12.LineDrawing(s0, s1, s2, x, y)
        # first order derivate
        s0_1df = [0, s0[0] * 3, s0[1] * 2, s0[2]]
        s1_1df = [0, s1[0] * 3, s1[1] * 2, s1[2]]
        s2_1df = [0, s2[0] * 3, s2[1] * 2, s2[2]]
        f1 = NCM12.LineDrawing(s0_1df, s1_1df, s2_1df, x, y)
        # second order derivate
        s0_2df = [0, 0, s0_1df[1] * 2, s0_1df[2]]
        s1_2df = [0, 0, s1_1df[1] * 2, s1_1df[2]]
        s2_2df = [0, 0, s2_1df[1] * 2, s2_1df[2]]
        f2 = NCM12.LineDrawing(s0_2df, s1_2df, s2_2df, x, y)
        return resolution, f0, f1, f2


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
        dmax, index = 0, 0
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


# https://ccjou.wordpress.com/2009/09/01/奇異值分解-svd/
class NCM09:
    def __init__(self, A, choice):
        self.A = A
        self.choice = choice
        "do something here"

    @staticmethod
    def diag_matrix_convert(v):
        # input L : vector v  ( v=[ 1 , 2, 3 , 4])
        # output diagonal matrix M with elements as 1,2,3,4 in diagonal
        i = v.shape[0]
        M = np.matrix(np.eye(i)) * 1.
        for index in range(0, i):
            M[index, index] = v[index]
        return M

    @staticmethod
    def power_method(A, choice):
        # input : matrix A, choice: 0:smallest; 1:biggest; 2:all eigenvalue
        # output : biggest eigenvalue
        i, j = A.shape[0], A.shape[1]
        if (i == j):
            result = 0
            if (choice == 0):
                result = NCM09.power_method_core(A.I)
                E = np.dot(A, result) / result[0]
                E = E[0]
                # print("smallest eigenvalue\n", result)
            elif (choice == 1):
                result = NCM09.power_method_core(A)
                E = np.dot(A, result) / result[0]
                E = E[0]
                # print("largest eigenvalue\n", result)
            elif (choice == 2):
                # print("all eigenvalues")
                result = NCM09.power_method_core(A)
                # print("largest eigenvalue\n", result)
                B, H = NCM09.Deflation_Householder_core(A, result)
                E = B
            else:
                print("invalid choice")
                E = 0
        else:
            print("not support in n!=m")
            E = 0

        return E

    @staticmethod
    def Deflation_Householder_core(A, E):
        # input : matrix A, found eigen vector E
        # output : a similar matrix H, where B=HAH^-1=HAH
        i, j = A.shape[0], A.shape[1]
        I = np.matrix(np.eye(i))
        seed = I[0:1, 0:j].T
        u = (E - seed) / NCM07.norm(E - seed)
        H = I - 2 * u * u.T
        B = H * A * H
        return B, H

    @staticmethod
    def power_method_core(A):
        # input : matrix A
        # output : biggest eigenvalue
        eps = np.spacing(1)  # Python precision in Gakki-machine
        i, j = A.shape[0], A.shape[1]
        if (i == j):
            u = A[0:i, 0:1]  # setup the initial vector u
            delta = 1
            count = 0
            while (delta > eps):
                uu = u
                try:
                    u = np.dot(A, u) / NCM07.norm(np.dot(A, u))
                    delta = abs(NCM07.norm(uu - u))
                    count = count + 1
                    if (count > 50):
                        # print("**** WARNING: duplicated eigenvalue or converge too slow ****")
                        pass
                        break
                except ValueError:
                    Error_Flg = True
                    print("Power Method fail")
                    break
        else:
            print("not support in n!=m")
            u = 1e-10

        return u

    @staticmethod
    def svd_qr(A):
        # input : matrix A
        # output :   u : { (..., M, M), (..., M, K) } array
        # output :   s : (..., K) array
        # output :   vh : { (..., N, N), (..., K, N) } array
        # SVD decompistion as : A=U*S*V.T
        # u, s, v = la.svd(A)
        # print("\n", u, "\n", s, "\n", v)
        i, j = A.shape[0], A.shape[1]
        ATA = np.dot(A.T, A)
        # E = NCM09.eigen_qr(ATA)
        EE, VV = la.eig(ATA)  # Python library
        # print("EE1 \n",EE)
        # print("VV1 \n", VV)
        idx = EE.argsort()[::-1]  # sorting the eigenvlaue
        EE = EE[idx]
        VV = VV[:, idx]
        u, s, v = A * 0., A * 0., A * 0.  # create the u,v all zero matrix
        EE = NCM09.diag_matrix_convert(EE) * 1.
        for v_index in range(0, i):
            index = i - v_index - 1
            s[v_index, v_index] = sqrt(EE[index, index])
            v[:, v_index] = VV[:, [index]]
            u[:, v_index] = np.dot(A, v[:, v_index]) / s[v_index, v_index]
        # print("SVD check2: ", NCM07.norm(A - np.dot(np.dot(u, s), v.T)))
        return u, s, v

    @staticmethod
    def eigen_qr(A):
        # input : matrix A
        # output :  eigenvalue of a symmetric real matrix by QR matrix
        eps = np.spacing(1)  # Python precision in Gakki-machine
        delta = 0
        while (abs(delta - A[0, 0]) > eps):
            Q, R, C = NCM08.gram_schmidt(A)
            delta = A[0, 0]
            A = np.dot(R, Q)
        R = np.diag(np.diag(A))
        return R

    @staticmethod
    def Exercise(A):
        # input : matrix A
        # output : Problem1 ~3   for copyright concern, the problems detail are not here.
        print("\n\nTarget Matrix:\n", A)
        i, j = A.shape[0], A.shape[1]
        if (i == j):
            print("Smallest Eigenvalue in Power Method :\n", NCM09.power_method(A, 0).round(3))
            print("Biggest Eigenvalue in Power Method :\n", NCM09.power_method(A, 1).round(3))
            print("Alll Eigenvalues in Power Method :\n", NCM09.power_method(A, 2).round(3))
            print("QR eigen :\n", NCM09.eigen_qr(A).round(3))
            u, s, v = NCM09.svd_qr(A)
            print("SVD decomposition A=U*S*V.T  U:\n", u.round(3))
            print("SVD decomposition A=U*S*V.T  S:\n", s.round(3))
            print("SVD decomposition A=U*S*V.T  V:\n", v.round(3))
            print("SVD decomposition check norm(A-U*S*V.T): ", NCM07.norm(A - np.dot(np.dot(u, s), v.T)))
        else:
            print("not support in this Matrix dimension")


class NCM08:
    def __init__(self):
        "do something here"

    # https://ccjou.wordpress.com/2010/04/22/gram-schmidt-正交化與-qr-分解/
    # https://en.wikipedia.org/wiki/QR_decomposition
    @staticmethod
    def gram_schmidt(A):
        # input : matrix A
        # output :  matrix Q, matrix R, matrix C ; where Q*R=A*C, permutation matrix
        # q, r = np.linalg.qr(A)  # Python Linalg formula
        U = NCM07.gram_schmidt(A)
        R = U * 0.  # create rrr space
        i, j = A.shape[0], A.shape[1]
        for j_index in range(0, j):
            for i_index in range(0, j_index + 1):
                R[i_index, j_index] = np.dot(np.transpose(U[:, i_index]), A[:, j_index])
        C = np.eye(i)  # no permutation process is done in this method
        # print("error : ", NCM07.norm(np.dot(U,R)-np.dot(A,C)))
        return (U, R, C)

    @staticmethod
    def gram_solver(A, b):
        # input : matrix A , vector b
        # output : solution vector x
        Q, R, C = NCM08.gram_schmidt(A)
        x = np.dot(np.dot(NCM05.Matrix_Inverse(R, np.eye(R.shape[0])), Q.T), b)
        # print("error : ", NCM07.norm(np.dot(A, x) - b))
        return x


class NCM07:
    def __init__(self):
        "do something here"

    # Norm = NCM07.norm(np.dot(np.dot(x.T, A), x))
    @staticmethod
    def gram_schmidt(A):
        # input : matrix A
        # output : matrix A, matrix U as the Gram_Schmidt matrix
        i, j = A.shape[0], A.shape[1]
        U = A * 0.  # create U ouput matrix
        P = np.zeros((i, j))
        I = np.matrix(np.eye((i)))
        for j_index in range(0, j):
            V = np.dot((I - P), A[0:i, j_index])
            U[0:i, j_index] = V / NCM07.norm(V)
            P = P + U[0:i, j_index] * U[0:i, j_index].T
        return (U)

    @staticmethod
    def gram_schmidt_check(U):
        # input : matrix U
        # output : column inner product. Zero means Correct !
        sum = 0
        i, j = U.shape[0], U.shape[1]
        sum = sum + np.dot(U[0:i, j - 1].T, U[0:i, 0])
        for j_index in range(0, j - 1):
            sum = sum + np.dot(U[0:i, j_index].T, U[0:i, j_index + 1])
        return sum

        # Norm = NCM07.norm(np.dot(np.dot(x.T, A), x))

    @staticmethod
    def norm(A):
        # input : matrix A
        # output : norm value
        sum = 0
        i, j = A.shape[0], A.shape[1]
        for i_index in range(0, i):
            for j_index in range(0, j):
                sum = sum + A[i_index, j_index] ** 2
        return (sqrt(sum))

    # https://ccjou.wordpress.com/2010/05/10/矩陣模/
    # https://ccjou.wordpress.com/2013/01/10/半正定矩陣的判別方法/
    # https://ccjou.wordpress.com/2013/01/07/答謝一誠──關於判定正定、負定或未定二次型的/
    @staticmethod
    def definite(A):
        # input : matrix A
        # output : value as the matrix type, 0: pos-def, 1: neg-def, 2:indef
        # Cholesky 分解：存在一 n\times n 階可逆矩陣 B，使得 A=B^TB (見“Cholesky 分解”)。
        Error_Flg = False
        L, Error_Flg = NCM07.chol(A)
        if (Error_Flg == False):
            type = 0  # pos-def
        else:
            L, Error_Flg = NCM07.chol(-A)
            if (Error_Flg == False):
                type = 1  # neg=def
            else:
                type = 2  # indef

        return type

    @staticmethod
    def choleski(A, b):
        # input : matrix A , vector x
        # output : solution
        # method : Ax=B  => A'Ax=A'b => LL'x=A'b => LZ=A'b,
        #       solve : Z by NCM06.forward(L, Pb) - L*Z=Pb
        #        => solve : Z=L'x  , x by backsubs(U, Z) - U*x=Z
        L, flag = NCM07.chol(np.dot(A.T, A))
        if (flag == False):
            Z = NCM06.forward(L, np.dot(A.T, b))  # LZ=(L'b)
            x = NCM06.backsubs(L.T, Z)  # L'x=Z
        else:
            print("Invalid matrix")
            x = 0
        return (x)

    # http://www.iiserpune.ac.in/~pgoel/LUDecomposition.pdf
    @staticmethod
    def forward(L, Pb):
        # forward elimination L*Z=Pb ,
        # input : L , Pb
        # output : Z
        i, j = L.shape[0], L.shape[1]
        print("L,Pb", L, Pb)
        Z = Pb * 0.  # create the Z report vector and force to float
        Z[0] = Pb[0, 0] / L[0, 0]
        for index_i in range(1, i):  # 1 to i
            Z[index_i] = (Pb[index_i] - np.dot(L[index_i, 0:index_i], Z[0:index_i, ])) / L[index_i, index_i]
        print("Z\n", Z)
        return Z

    @staticmethod
    def chol(A):
        # input matrix A
        # output matrix L as the result of the cholesky matrix
        #  Python module : L = np.linalg.cholesky(A)
        Matrix = A
        i, j = Matrix.shape[0], Matrix.shape[1]  # parse the matrix dimension information
        r_matrix = np.zeros((i, j))
        r_matrix[0, 0] = 1. * Matrix[0, 0] ** 0.5
        for i_index in range(1, i):  # setup the seed in the [0, ] location
            r_matrix[i_index, 0] = Matrix[i_index, 0] / r_matrix[0, 0]
        for i_index in range(1, i):  # process the each element
            for j_index in range(1, j - (i - i_index) + 1):
                try:
                    r_matrix[i_index, j_index] = NCM07. \
                        c_iteration(Matrix[i_index, j_index], i_index, j_index, r_matrix)
                    Error_Flg = False
                    if (np.isfinite(r_matrix[i_index, j_index]) != True):
                        Error_Flg = True
                        break
                except ValueError:  # not possible to perform the Cholesky !!
                    Error_Flg = True
        if Error_Flg == True:
            # r_matrix=0
            pass
        return (r_matrix, Error_Flg)

    @staticmethod
    def c_iteration(input, i, j, r_matrix):
        # input each element
        # output L-element value by iteration
        result = input
        for j_index in range(0, j):
            result = (result - (r_matrix[i, j_index]) * (r_matrix[j, j_index]))
        if (i == j):
            result = sqrt(result)
        else:
            result = result / r_matrix[i - 1, j]
        return (result)


class NCM06:
    # https://blog.csdn.net/u013088062/article/details/50353202
    # https://stackoverflow.com/questions/37515461/sorting-all-rows-in-numpy-matrix-by-target-column

    def __init__(self):
        pass

    def lu_colx(Matrix):
        AllMatrix = Matrix * 1.  # force to float format in Python
        i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
        U_Matrix = AllMatrix * 0.  # create the result matrix with all zero with float format
        L_Matrix = np.matrix(np.eye(j)) * 1.  # create the L_matrix and force in the float
        P_Matrix = np.matrix(np.eye(j)) * 1.
        SubMatrix = AllMatrix
        SubMatrix, P_original = NCM06.Matrix_sort(SubMatrix)  # derive the P_orignal vector
        for index in range(0, i):
            SubMatrix, Vector, P, L_vector = NCM06.Row_handle(SubMatrix)
            U_Matrix[index, index:j] = Vector
            L_Matrix[index:i, index] = L_vector + L_Matrix[index:i, index]
            if (L_Matrix[index + 1:i, 0:index + 1].shape[0] > 1.99):  # start the P-checking
                L_Matrix[index + 1:i, 0:index + 1] = NCM06.L_Matrix_handle(L_Matrix[index + 1:i, 0:index + 1], P)
            if (index < i - 2):  # stop P matrix management in the too small case for P.shape issue
                P_original = NCM06.P_handle(P_original, P)
        P_Matrix = NCM06.P_parse(np.matrix(np.eye(j)) * 0., P_original)
        return (L_Matrix, U_Matrix, P_Matrix)

    def P_handle(P_original, P):
        # input P_original , P vector from sort
        # output P_original with the P vector fix
        j = P_original.shape[0]
        k = P.shape[0]
        P_Vector = np.zeros((k, k))
        for index in range(0, k):
            key = P[index]
            P_Vector[index, key] = 1
        P_original[j - k:j] = np.dot(P_original[j - k:j], P_Vector)
        return (P_original)

    def Row_handle(Matrix):
        # 　 input : Matrix
        # 　 output : SubMatrix, Display-Vector, P, L_vector
        i, j = Matrix.shape[0], Matrix.shape[1]
        P = 0
        Matrix = Matrix * 1.  # force to change to float mode
        L_vector = np.matrix(np.zeros(j)).T
        for index in range(0, i - 1):
            L_factor = (Matrix[index + 1, 0] / Matrix[0, 0])
            Matrix[index + 1,] = -L_factor * Matrix[0,] + Matrix[index + 1,]
            L_vector[index + 1, 0] = L_factor
        SubMatrix = Matrix[1:i, 1:j]
        if (i > 1):  # stop the sort as dim of matrix is too small
            SubMatrix, P = NCM06.Matrix_sort(SubMatrix)  # lift up the bigger pivot and report P
        else:
            pass
        return (SubMatrix, Matrix[0,], P, L_vector)

    def L_Matrix_handle(L_Matrix, P):
        # input L_matrix , P vector from sort
        # output L_Matrix with the P vector fix
        pp = P.shape[0]
        I = np.matrix(np.zeros((pp, pp)))
        for index in range(0, pp):
            I[index, P[index]] = 1
        L_Matrix = I * L_Matrix
        return (L_Matrix)

    def P_parse(P_Matrix, P):
        # input P_Matrix , P vector from sort
        # output P_Matrix as the final
        k = P.shape[0]
        for index in range(0, k):
            P_Matrix[index, P[index]] = 1
        return (P_Matrix)

    def Matrix_sort(Matrix):
        # Input: A n by n real matrix, index for comparison.
        # Output: Sorted Matrix , P_index for P
        P_index = np.argsort(-Matrix.A[:, 0])  # - : reversed order
        Sorted_Matrix = Matrix[P_index]
        return (Sorted_Matrix, P_index)  # P_index is used for Permutation Matrix

    def lu_colx_SciPY(A):
        # Input: A n by n real matrix
        # Output: L low triangular matrix , U upper-triangular matrix
        #              Q column permutation matrix ( or permutation vector)
        AllMatrix = A
        P, L, U = la.lu(AllMatrix)
        # print("Python Scipy result P: \n", P)
        # print("Python Scipy result L: \n", L)
        # print("Python Scipy result U: \n", U)
        # print("Python Scipy result check: \n", np.dot(L,U)-np.dot(P,A))

    import numpy as np

    # method in the LU decomposition
    # AX=b  => P*AX=P*b  => PA*X=Pb, PA=LU => LU*X=Pb  ...fist step
    # L*(UX)=Pb => find  UX=L^-1*Pb  => solve X=U^-1*(L^-1*P*b)
    # deinition : z=UX ; b^=Pb
    def Axb_solver(A, b):
        # Problem 2:
        # input : A,b  (A*x=b)
        # output : x by LU decomposition method
        L, U, P = NCM06.lu_colx(A)
        Pb = np.dot(P, b)
        Z = NCM06.forward(L, Pb)
        X = NCM06.backsubs(U, Z)
        return (X)

    # http://www.iiserpune.ac.in/~pgoel/LUDecomposition.pdf
    @staticmethod
    def forward(L, Pb):
        # forward elimination L*Z=Pb ,
        # input : L , Pb
        # output : Z
        i, j = L.shape[0], L.shape[1]
        # print("L,Pb",L,Pb)

        Z = Pb * 0.  # create the Z report vector and force to float
        Z[0] = Pb[0, 0] / L[0, 0]
        for index_i in range(1, i):  # 1 to i
            Z[index_i] = (Pb[index_i] - np.dot(L[index_i, 0:index_i], Z[0:index_i, ])) / L[index_i, index_i]
        # print("Z\n", Z)
        return Z

    @staticmethod
    def backsubs(U, Z):
        # back substitution U*x=Z
        # input : U,Z
        # output : X
        i, j = U.shape[0], U.shape[1]
        X = Z * 0.  # create the X report vector and force to float
        X[i - 1] = Z[i - 1] / U[i - 1, j - 1]
        for index_j in range(j - 1, 0, -1):  # backward loop. must be careful !
            X[index_j - 1] = (Z[index_j - 1] - (np.dot(U[index_j - 1, index_j:j], \
                                                       X[index_j:j, ]))) / U[index_j - 1, index_j - 1]
        return (X)

    @staticmethod
    def Matrix_inverse(A):
        # input : Matrix A
        # output : X
        # X should be the inversed matrix of input Matrix A
        # expectation : Call lu_colx(A) => LZ=I => UX^=Z => X=P*X^
        L, U, P = NCM06.lu_colx(A)  # AX=I => LUX=P*I => LUX=P
        Z = NCM06.forward(L, P)  # LUX=P => LZ=P => find Z
        X = NCM06.backsubs(U, Z)  # UX=Z => find X equal A^-1
        return (X)

        # https://keisan.casio.com/exec/system/15076953047019
        #

    import warnings
    warnings.filterwarnings("ignore")


class NCM05:
    def __init__(self):
        pass

    # GaussElimination
    @staticmethod
    def Gauss_eimination(A, b):
        AllMatrix = np.concatenate((A, b), axis=1) * 1.0  # force to float format in Python

        i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
        Result_Matrix = AllMatrix - AllMatrix  # create the result matrix with all zerp
        temp = []  # empty list
        SubMatrix = AllMatrix  # for initial Matrix
        index = 0
        for index in range(0, j - 1):
            Vecotr, SubMatrix = NCM05.Gauss_approach(SubMatrix)  # start Vector and SubMatrix calculation
            Vecotr = np.append(temp, Vecotr)  # install the zero on prefix elements
            Result_Matrix[index,] = Vecotr  # rebuild the Result Matrix
            temp.append(0)  # added the zero
        return (Result_Matrix)

    @staticmethod
    def Gauss_approach(Matrix):
        i, j = Matrix.shape[0], Matrix.shape[1]  # parse the matrix dimension information
        # print(Matrix)
        for ii in range(0, i):  # alignment with all first numbers
            Matrix[ii] = Matrix[ii] / Matrix[ii, 0]  # unity the Pivot
        Matrix = np.sort(Matrix, axis=0)  # pivot move up to first row line
        # angela = 0  # i miss you after 1998 graduation
        for angela in range(0, i - 1):
            Matrix[angela + 1] = Matrix[angela + 1] - Matrix[0]
            # start the Gauss Elimination process
        Vector = Matrix[0,]  # extract the output message line
        SubMatrix = Matrix[1:j, 1: i + 1]  # prepare the next calculation portion
        return (Vector, SubMatrix)

    # GaussJordan , input: Gauss_eimination Matrix; output: Gauss_Jordan Matrix
    @staticmethod
    def GaussJordan(A, b):
        AllMatrix = np.concatenate((A, b), axis=1) * 1.0  # force to float format in Python
        i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
        SubMatrix = AllMatrix
        for ii in range(0, i):
            SubMatrix = NCM05.GaussJordan_approach(ii, SubMatrix)
        AllMatrix = np.matrix(np.eye(i) * 1)
        AllMatrix = np.append(AllMatrix, SubMatrix, axis=1)
        return (AllMatrix)

    @staticmethod
    def GaussJordan_approach(index, Matrix):
        i, j = Matrix.shape[0], Matrix.shape[1]  # parse the matrix dimension information
        # print("Matrix[index, 0]  ::",Matrix[index, 0])
        if (Matrix[index, 0] == 0):
            # print("NO SOLUTION")
            pass
        else:
            Matrix[index,] = Matrix[index,] / Matrix[index, 0]  # unity the first row
            for ii in range(0, i):
                if (ii != index):
                    Matrix[ii,] = (Matrix[index,] * Matrix[ii, 0] * -1) + Matrix[ii,]
                else:
                    Matrix[ii,] = Matrix[index,]
            SubMatrix = Matrix[0:i, 1:j]
        return (SubMatrix)

    # Matrix Inversion
    @staticmethod
    def Matrix_Inverse(A, I):
        i, j = I.shape[0], I.shape[1]  # parse the matrix dimension information
        AllMatrix = NCM05.GaussJordan(A, I)
        AllMatrix = AllMatrix[0:i, i:2 * i]
        return (AllMatrix)

    @staticmethod
    def rref_solver(A, b):
        AllMatrix = np.concatenate((A, b), axis=1) * 1.0  # force to float format in Python
        i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
        free_variable = np.ones((j - 1, 1))
        try:
            AllMatrix = NCM05.GaussJordan(A, b)
            root_vector = AllMatrix[0:i, j - 1:j]
            if (len(root_vector) == (j - 1)):
                status = 1  # unique solution
            else:
                for ff in range(0, len(root_vector)):
                    free_variable[ff, 0] = root_vector[ff] - 1
                root_vector = np.vstack([root_vector, 0])
                status = 2  # multi solutions
        except UnboundLocalError:
            root_vector = 0
            status = 3  # no solution
        return (status, root_vector, free_variable)


import matplotlib.pyplot as plt
import random


def main():
    print("let's make it happen")

    x = [0, 1, 2, 3]
    y = [0, 0.5, 2.0, 1.5]

    # 321
    plt.subplot(161)
    x, y, res, f = NCM12.Problem1(x, y)

    plt.plot(x, y, color='cyan', linewidth=4.0, marker='d', markevery=1, label='f-Python')
    plt.plot(res, f, color='green', markevery=5, marker=9, linestyle='--', label='f-own ')
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0, 3])
    axes.set_ylim([-5, 5])
    plt.gca().set_xticks(np.arange(0, 3.1, 1))
    plt.gca().set_yticks(np.arange(-5, 5, 1))
    plt.title(' A linear Spline ')
    plt.grid(True)

    # 322
    plt.subplot(162)
    res, cs, cs_1, cs_2, f, f_1, f_2 = NCM12.Problem2_curve1_clamped_cubic(x, y, 0.2, -1)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=2.0, marker='d', markevery=1,
             label='Original')
    plt.plot(res, cs, color='pink', marker=9, linewidth=4.0, markevery=5, label=' f-scipy ')
    plt.plot(res, cs_1, color='gray', marker=10, linewidth=4.0, markevery=5, label=' f`-scipy ')
    plt.plot(res, cs_2, color='cyan', marker=6, linewidth=4.0, markevery=5, label=' f``-scipy ')
    plt.plot(res, f, color='blue', marker=6, linestyle='-', markevery=6, label=' f-own ')
    plt.plot(res, f_1, color='black', marker=10, linestyle='-.', markevery=8, label=' f`-own ')
    plt.plot(res, f_2, color='red', marker=6, linestyle='--', markevery=10, label=' f``-own ')
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0, 3])
    axes.set_ylim([-5, 5])
    plt.gca().set_xticks(np.arange(0, 3.1, 1))
    plt.gca().set_yticks(np.arange(-5, 5, 1))
    plt.title(' Clamped Spline ')

    plt.grid(True)

    # 323
    plt.subplot(163)
    res, cs, cs_1, cs_2, f, f_1, f_2 = NCM12.Problem2_curve2_natural(x, y, 0, 0)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=2.0, marker='d', markevery=1,
             label='Original')
    plt.plot(res, cs, color='pink', marker=9, linewidth=4.0, markevery=5, label=' f-scipy ')
    plt.plot(res, cs_1, color='gray', marker=10, linewidth=4.0, markevery=5, label=' f`-scipy ')
    plt.plot(res, cs_2, color='cyan', marker=6, linewidth=4.0, markevery=5, label=' f``-scipy ')
    plt.plot(res, f, color='blue', marker=6, linestyle='-', markevery=6, label=' f-own ')
    plt.plot(res, f_1, color='black', marker=10, linestyle='-.', markevery=8, label=' f`-own ')
    plt.plot(res, f_2, color='red', marker=6, linestyle='--', markevery=10, label=' f``-own ')
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0, 3])
    axes.set_ylim([-5, 5])
    plt.gca().set_xticks(np.arange(0, 3.1, 1))
    plt.gca().set_yticks(np.arange(-5, 5, 1))
    plt.title(' Natural Spline ')

    plt.grid(True)

    # 324
    plt.subplot(164)
    res, cs, cs_1, cs_2, f, f_1, f_2 = NCM12.Problem2_curve2_NotAKnot(x, y)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=2.0, marker='d', markevery=1,
             label='Original')
    plt.plot(res, cs, color='pink', marker=9, linewidth=4.0, markevery=5, label=' f-scipy ')
    plt.plot(res, cs_1, color='gray', marker=10, linewidth=4.0, markevery=5, label=' f`-scipy ')
    plt.plot(res, cs_2, color='cyan', marker=6, linewidth=4.0, markevery=5, label=' f``-scipy ')
    plt.plot(res, f, color='blue', marker=6, linestyle='-', markevery=6, label=' f-own ')
    plt.plot(res, f_1, color='black', marker=10, linestyle='-.', markevery=8, label=' f`-own ')
    plt.plot(res, f_2, color='red', marker=6, linestyle='--', markevery=10, label=' f``-own ')
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0, 3])
    axes.set_ylim([-5, 5])
    plt.gca().set_xticks(np.arange(0, 3.1, 1))
    plt.gca().set_yticks(np.arange(-5, 5, 1))
    plt.title(' Not-a-Knot Spline ')

    plt.grid(True)

    # 325
    plt.subplot(165)
    res, f, f_1, f_2 = NCM12.Problem2_curve2_ParabolicallyTerminate(x, y)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=2.0, marker='d', markevery=1,
             label='Original')
    # plt.plot(res, cs, color='pink', marker=9, linewidth=4.0, markevery=5, label=' f ')
    # plt.plot(res, cs_1, color='gray', marker=10, linewidth=4.0, markevery=5, label=' f` ')
    # plt.plot(res, cs_2, color='cyan', marker=6, linewidth=4.0, markevery=5, label=' f`` ')
    plt.plot(res, f, color='blue', marker=6, linestyle='-', markevery=6, label=' f-own ')
    plt.plot(res, f_1, color='black', marker=10, linestyle='-.', markevery=8, label=' f`-own ')
    plt.plot(res, f_2, color='red', marker=6, linestyle='--', markevery=10, label=' f``-own ')
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0, 3])
    axes.set_ylim([-5, 5])
    plt.gca().set_xticks(np.arange(0, 3.1, 1))
    plt.gca().set_yticks(np.arange(-5, 5, 1))

    plt.title(' Parabolically \n' + ' No Python code')
    plt.grid(True)

    # 326
    plt.subplot(166)
    res, f, f_1, f_2 = NCM12.Problem2_curve2_EndpointCurvatureAdjustSpline(x, y, -0.3, 3.3)
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=2.0, marker='d', markevery=1,
             label='Original')
    # plt.plot(res, cs, color='pink', marker=9, linewidth=4.0, markevery=5, label=' f ')
    # plt.plot(res, cs_1, color='gray', marker=10, linewidth=4.0, markevery=5, label=' f` ')
    # plt.plot(res, cs_2, color='cyan', marker=6, linewidth=4.0, markevery=5, label=' f`` ')
    plt.plot(res, f, color='blue', marker=6, linestyle='-', markevery=6, label=' f-own ')
    plt.plot(res, f_1, color='black', marker=10, linestyle='-.', markevery=8, label=' f`-own ')
    plt.plot(res, f_2, color='red', marker=6, linestyle='--', markevery=10, label=' f``-own ')
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0, 3])
    axes.set_ylim([-5, 5])
    plt.gca().set_xticks(np.arange(0, 3.1, 1))
    plt.gca().set_yticks(np.arange(-5, 5, 1))

    plt.title(' Endpoint curvature-adjust\n' + ' Spline \n' + ' No Python code')
    plt.grid(True)
    plt.show()

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
    if False:

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
