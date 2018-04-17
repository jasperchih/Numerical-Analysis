#   # -*- coding: UTF-8 -*-
#   trial on the : Satomi machine
#   Created by Ush on 2018/3/16
#   Project name :  Class3_FindingRoots 
#   Please contact CHIH, HSIN-CHING/D0631008 when expect to refer this source code.
#   NOTE : no liability on any loss nor damage by using this source code. it is your own risk.

import numpy as np
import math


def FUNC(type, x):
    F_select = 0
    if (type == 1):  # function1 is f(x)=x^2-1
        F_select = x ** 2 - 2
    elif (type == 2):
        F_select = x ** 5 + x ** 3 + 3
    elif (type == 3):
        F_select = x ** 3 - 2 * math.sin(x) - 1
    elif (type == 4):
        F_select = 0.5 * math.tan(x) - math.tanh(x)
    else:
        print("invalid function")
    return (float(F_select))


def Bisection(f, left, right, eps):
    iteration = 0
    while (math.fabs(right - left) > eps * math.fabs(right)):
        middle = (left + right) / 2
        if (np.sign(FUNC(f, middle)) == np.sign(FUNC(f, right))):
            right = middle
        else:  # root is  founded, move left = middle value
            left = middle
        iteration = iteration + 1
        # print(iteration, "\t", right, "\t", FUNC(f, right))
    return (right, iteration)


def Secant(f, left, right, eps):
    iteration = 0
    while (math.fabs(right - left) > eps * math.fabs(right)):
        middle, left = left, right
        right = right + ((right - middle) / ((FUNC(f, middle)) / (FUNC(f, right)) - 1))
        iteration = iteration + 1
        # print(iteration, left, FUNC(f, left), right, FUNC(f, right))
    return (right, iteration)


# IQI : Inverse Quadratic Interpolation
# https://en.wikipedia.org/wiki/Inverse_quadratic_interpolation
# https://nickcdryan.com/2017/09/13/root-finding-algorithms-in-python-line-search-bisection-secant-newton-raphson-boydens-inverse-quadratic-interpolation-brents/
def PolyInterp(f, p0, p1, p2):
    a, b, c = FUNC(f, p0), FUNC(f, p1), FUNC(f, p2)
    L0 = (p0 * b * c) / ((a - b) * (a - c))
    L1 = (p1 * a * c) / ((b - a) * (b - c))
    L2 = (p2 * b * a) / ((c - a) * (c - b))
    return (L0 + L1 + L2)


def IQI(f, left, middle, right, eps):
    iteration = 0
    while (math.fabs(right - middle) > eps * math.fabs(right)):
        x = PolyInterp(f, left, middle, right)
        left, middle, right = middle, right, x
        iteration = iteration + 1
        # print(iteration, right, FUNC(f, right))
    return (right, iteration)


# https://en.wikipedia.org/wiki/Brent%27s_method#Dekker.27s_method
# it is Dekker's method in 1969
# https://blogs.mathworks.com/cleve/2015/10/12/zeroin-part-1-dekkers-algorithm/
# https://cocalc.com/share/ab93d447b6728ae561bf1f9e18f0b103316d715f/Efficiency%20of%20Standard%20and%20Hybrid%20Root%20Finding%20Methods.sagews?viewer=share
def Mixed_Scant_Bisection(f, left, right, eps):
    fa = FUNC(f, left)
    fb = FUNC(f, right)
    if (np.sign(fa) == np.sign(fb)):
        print(" Interval in ", left, " and ", right, " is in the same sign signal")
    #print("initial-value from :", right, "\t", fb)
    #print("N-Type-Step\t\tbn\t\t\t\t\t\tf(bn) ")
    iteration = 1
    # left is the previous value of right and [right, middle] always contains the zero.
    middle, fc = left, fa
    while (math.fabs(right - left) > eps):
        if (np.sign(fb) == np.sign(fc)):
            middle, fc = left, fa
        if (math.fabs(fc) < math.fabs(fb)):  # Swap to insure f(b) is the smallest value so far.
            left, fa = right, fb
            right, fb = middle, fc
            middle, fc = left, fa
        mid_point = float(right + middle) / 2  # BiSection ( Step1 )
        # secant_left/secant_right is the the secant step.
        secant_left = (right - left) * fb
        if (secant_left >= 0):
            secant_right = fa - fb  # swap like BiSection
        else:
            secant_right = -1 * (fa - fb)  # swap like BiSection
            secant_left = -1 * secant_left
        left, fa = right, fb  # prepare for the next iteration process points.
        if (secant_left <= ((mid_point - right) * secant_right)):
            right = right + (secant_left / secant_right)  # Secant
            fb = FUNC(f, right)
            #print(iteration, "Secant-step", right, "\t", fb)
        else:
            right = mid_point  # BiSection
            fb = FUNC(f, right)
            #print(iteration, "Bisect-step", right, "\t", fb)
        iteration = iteration + 1
    return (right, iteration)


# https://cocalc.com/share/ab93d447b6728ae561bf1f9e18f0b103316d715f/Efficiency%20of%20Standard%20and%20Hybrid%20Root%20Finding%20Methods.sagews?viewer=share
def brentsMethod(f, a, b, accuracy):
    '''
    Code inspired by:
    https://en.wikipedia.org/wiki/Brent's_method (The pseudocode was very helpful in translating this to Python)
    http://blogs.mathworks.com/cleve/2015/10/26/zeroin-part-2-brents-version/
    Function that computes an approximate root
    for a given function using Dekker's method.
    args:
        f: a function
        a, b: an initial value, most efficient when close to the root
        iterations: number of times to run the loop
    output:
        a list that contains an approximate root s and the number
        of iterations required
    '''
    fa = FUNC(f, a)
    fb = FUNC(f, b)
    # When this is set, we use the bisection method
    bisection = True
    if fa * fb >= 0:
        raise Exception("Invalid inputs for a and b, require sign change")
    c = a
    s = 0
    d = 0
    iteration = 0
    while abs(b - a) > accuracy:
        # quadratic interpolation
        if (FUNC(f, a) != FUNC(f, c)) and (FUNC(f, b) != FUNC(f, c)):
            s = PolyInterp(f, a, b, c)      # IQI core
            #print(iteration, "IQI-step", s, "\t", FUNC(f,s))
        else:
            s = ((a * FUNC(f, b)) - (b * FUNC(f, a))) / (FUNC(f, b) - FUNC(f, a))# secant
            #print(iteration, "Secant-step", s, "\t", FUNC(f, s))
        if brentConditional(s, a, b, c, d, bisection, accuracy):
            s = (a + b) / 2.0 # bisection
            bisection = True
        else:
            bisection = False
        d = c
        c = b
        if (FUNC(f, a) * FUNC(f, s) < 0):   #step (3a)
            b = s
            fb = FUNC(f, s)
        else:
            a = s
            fa = FUNC(f, s)
        if abs(fa) < abs(fb):
            temp = a
            a = b
            b = temp
            temp2 = fa
            fa = fb
            fb = temp2
        iteration = iteration + 1
    return (s, iteration)


def brentConditional(s, a, b, c, d, mflag, accuracy):
    if ((s < (3 * a + b) * 0.25) or
            (mflag and (abs(s - b) >= (abs(b - c) * 0.5)) or
                 (not mflag and (abs(s - b) >= (abs(c - d) * 0.5)) or
                      (mflag and (abs(b - c) < accuracy)) or
                      (not mflag and (abs(c - d) < accuracy))))):
        return True
    return False


import matplotlib.pyplot as plt
from scipy import optimize


def f1(x):
    return (x ** 2 - 2)


def main():
    print
    "let's make it happen!"
    # eps function in matlab
    # Python code expression : np.spacing(1)
    eps = np.spacing(1)
    eps = 1e-6
    print("Python eps : " + str(eps))

    # formula 0
    print("\nFormula 0 : x ** 2 - 2 \t\t\t (root, iteration times)")
    ITR, Root = Bisection(1, 1, 2, eps)
    print("BiSection Result:\t\t\t\t", Bisection(1, 1, 2, eps))
    print("Secant Result:\t\t\t\t\t", Secant(1, 1, 2, eps))
    print("IQI Result:\t\t\t\t\t\t", IQI(1, 1, 1.5, 2, eps))
    print("Mixed_Scant_Bisection Result:\t", Mixed_Scant_Bisection(1, 1, 2, eps))
    print("Brent Result:\t\t\t\t\t", brentsMethod(1, 1, 2, eps))

    # formula 1
    print("\nFormula 1 : x ** 5 + x ** 3 + 3  (root, iteration times)")
    print("BiSection Result:\t\t\t\t", Bisection(2, -2, 0, eps))
    print("Secant Result:\t\t\t\t\t", Secant(2, -2, 0, eps))
    print("IQI Result:\t\t\t\t\t\t", IQI(2, -2, -1, 0, eps))
    print("Mixed_Scant_Bisection Result:\t", Mixed_Scant_Bisection(2, -2, 0, eps))
    print("Brent Result:\t\t\t\t\t", brentsMethod(2, -2, 0, eps))

    # formula 2
    print("\nFormula 2 : x ** 3 - 2 * math.sin(x) - 1\t  (root, iteration times)")
    print("BiSection Result:\t\t\t\t", Bisection(3, 0, 2, eps))
    print("Secant Result:\t\t\t\t\t", Secant(3, 0, 2, eps))
    print("IQI Result:\t\t\t\t\t\t", IQI(3, 0, 1, 2, eps))
    print("Mixed_Scant_Bisection Result:\t", Mixed_Scant_Bisection(3, 0, 2, eps))
    print("Brent Result:\t\t\t\t\t", brentsMethod(3, 0, 2, eps))

    # formula 3
    print("\nFormula 3 : 0.5 * math.tan(x) - math.tanh(x)\t  (root, iteration times)")
    A = 1e-3
    B = math.pi / 2
    print("BiSection Result:\t\t\t\t", Bisection(4, A, B, eps))
    print("Secant Result:\t\t\t\t\t", Secant(4, A, B, eps))
    print("IQI Result:\t\t\t\t\t\t", IQI(4, A, B, B, eps))
    print("Mixed_Scant_Bisection Result:\t", Mixed_Scant_Bisection(4, A, B, eps))
    print("Brent Result:\t\t\t\t\t", brentsMethod(4, A, B, eps))


if __name__ == "__main__":
    main()
