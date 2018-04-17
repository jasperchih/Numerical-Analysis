#   # -*- coding: UTF-8 -*-
#   trial on the : \TBD
#   Created by Gakki on 2018/3/11
#   Project name :  class 2 - Polynomial in Lagrange, Newton, and Bezier methods
#   Please contact CHIH, HSIN-CHING/D0631008 when expect to refer this source code.
#   NOTE : no liability on any loss nor damage by using this source code. it is your own risk.




import time
import numpy
import matplotlib.pyplot as plt

# reference Link:
# https://gist.github.com/aurelienpierre/1d9826e7db078e048bf437e516a7a4b2
# Input : a numpy nx2 array of the interpolation points
# Return : 1. Symbolic Expression
#          2. Polynomial equation result



from sympy import *
from sympy.matrices import *
import numpy
import pylab
import warnings
import numpy as np
import numpy as np
from scipy.misc import comb

def Lagrange_interpolation(x, y, u):
    """
    Compute the Lagrange interpolation polynomial.
    # points : input x and y array
    # u : return value
    :var points: A numpy n×2 ndarray of the interpolations points
    :var variable: None, float or ndarray
    :returns:   * P the symbolic expression
                * Y the evaluation result of the polynomial  
    : Equation as : Pn(x)=sum ( Li(x)*yi ), n =0 ~ n ;     
    """
    Numbers = np.size(x)  # detect for how many input points
    dimension = np.size(u)
    report_vector = []  # create the report vectors for input u vector space
    for index in range(0, dimension):
        calculation = u[index]  # feed the input value
        result = 0  # clean the every input value result
        for i in range(0, Numbers):
            numerator = 1  # reset the numerator value on each line
            denominator = 1  # reset the denominator value on each line
            for j in range(0, Numbers):
                if (j != i):
                    numerator = numerator * (calculation - x[0][j])
                    denominator = denominator * (x[0][i] - x[0][j])
                else:
                    print("not process while i=j")
            result = result + y[0][i] * (numerator / denominator)
        report_vector = np.append(report_vector, result)
    return report_vector
def Newton_interpolation(x, y, u):
    """
    reference :
    https://stackoverflow.com/questions/14823891/newton-s-interpolating-polynomial-python
    """
    x.astype(float)
    y.astype(float)
    n = np.size(x)
    Cn_parameters = []
    report_vector = []
    for i in range(0, n):
        Cn_parameters.append(y[0][i])
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            Cn_parameters[i] = float(Cn_parameters[i] - Cn_parameters[i - 1]) \
                               / float(x[0][i] - x[0][i - j])  # store the Cn parameters
    # print("The Cn Parameters = " + str(Cn_parameters))
    # setup the value calculation routine
    for i in range(0, np.size(u)):
        n = len(Cn_parameters) - 1
        result = Cn_parameters[n]
        for j in range(n - 1, -1, -1):
            # formula=C0+C1(u-x0)+C2(u-x0)(u-x1).....+Cn(u-x0)..(u-x_n-1)
            result = result * (u[i] - x[0][j]) + Cn_parameters[j]
            print(n - 1)
        report_vector = np.append(report_vector, result)

    return report_vector
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    # comb(n,i) : represents the n chooses it.
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
def bezier_curve(nPoints, nTimes):
    points = np.random.rand(nPoints, nTimes)  # create the matrix with nPoints x nTimes size
    xpoints = [p[0] for p in points]   # choose the p[0] as the starting point
    ypoints = [p[1] for p in points]   # chosse the p[0] as the ending point
    u = np.linspace(0.0, 1.0, nTimes)   # setup the interval as u vector 1 x n
    # core of the Bezier function calculaton upon the control point ( nPoints) and u vector
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, u) for i in range(0, nPoints)])
    xvals = np.dot(xpoints, polynomial_array)
    yvals = np.dot(ypoints, polynomial_array)
    return points, xpoints, ypoints, xvals, yvals

def main():
    print
    "let's make it happen!"

    # setup the array of the points
    TestCluster1X = numpy.array([
        [0, 2, 3, 4]
    ])
    TestCluster1Y = numpy.array([
        [7, 11, 28, 63]
    ])

    TestCluster2X = numpy.array([
        [1, 2, 3, 4, 5, 6]
    ])
    TestCluster2Y = numpy.array([
        [16, 18, 21, 17, 15, 12]
    ])

    u1 = np.linspace(TestCluster1X[0][0], TestCluster1X[0][-1], 100)  # -1 represent the end
    u2 = np.linspace(TestCluster2X[0][0], TestCluster2X[0][-1], 100)  # -1 represent the end

    # Task1 : Lagrange Method
    if True:
        P_Result1 = Lagrange_interpolation(TestCluster1X, TestCluster1Y, u1)
        P_Result2 = Lagrange_interpolation(TestCluster2X, TestCluster2Y, u2)
        f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        ax1.annotate('approach curve', xy=(3.2, 30), xytext=(3.5, 15), arrowprops=dict(shrink=0.1))
        ax1.plot(TestCluster1X, TestCluster1Y, 'D', color='r', label='original coordinates')
        ax1.plot(u1, P_Result1, '.', color='g', label='Lagrange method result')
        ax2.plot(TestCluster2X, TestCluster2Y, 'D', color='r', label='original coordinates')
        ax2.plot(u2, P_Result2, '.', color='g', label='Lagrange method result')
        plt.show()

    # Task 2 : Newton Method
    if True:
        P_Result3 = Newton_interpolation(TestCluster1X, TestCluster1Y, u1)
        P_Result4 = Newton_interpolation(TestCluster2X, TestCluster2Y, u2)
        f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        ax1.annotate('Newton Method curve', xy=(2.8, 25), xytext=(3.0, 10), arrowprops=dict(shrink=0.1))
        ax1.plot(TestCluster1X, TestCluster1Y, 'D', color='r', label='original coordinates')
        ax1.plot(u1, P_Result3, '.', color='g', label='Newton method result')
        ax2.plot(TestCluster2X, TestCluster2Y, 'D', color='r', label='original coordinates')
        ax2.plot(u2, P_Result4, '.', color='g', label='Newton method result')

        plt.show()

    # Task 3 : Bezier function
    if True:
        # control points, u-array
        # modification : https://stackoverflow.com/questions\
        #  /12643079/b%C3%A9zier-curve-fitting-with-scipy
        # points : x,y coordinate of control points
        # xpoints,ypoints :
        # xvals, yvals : bezier curve from starting point to ending points
        # nPoints : numbers of control points
        # u : numbers of the bezier curve
        nPoints = 6
        u = 80
        points, xpoints, ypoints, xvals, yvals = bezier_curve(nPoints, u)
        plt.plot(xvals, yvals)  # draw out the trace between control points
        plt.plot(xpoints, ypoints, "r*")  # draw out the nPoints ( include control points )
        for nr in range(len(points)):
            plt.text(points[nr][0], points[nr][1], nr)  # place the 0, 1, 2 .. next to points
        plt.show()


if __name__ == "__main__":
    main()
