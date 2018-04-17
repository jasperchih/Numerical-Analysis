#   # -*- coding: UTF-8 -*-
#   trial on the : Satomi machine
#   Created by Ush on 2018/3/30
#   Project name :  Class05 
#   Please contact CHIH, HSIN-CHING/D0631008 when expect to refer this source code.
#   NOTE : no liability on any loss nor damage by using this source code. it is your own risk.

from __future__ import division


# GaussElimination
def Gauss_eimination(A, b):
    AllMatrix = np.concatenate((A, b), axis=1) * 1.0  # force to float format in Python

    i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
    Result_Matrix = AllMatrix - AllMatrix  # create the result matrix with all zerp
    temp = []  # empty list
    SubMatrix = AllMatrix  # for initial Matrix
    index = 0
    for index in range(0, j - 1):
        Vecotr, SubMatrix = Gauss_approach(SubMatrix)  # start Vector and SubMatrix calculation
        Vecotr = np.append(temp, Vecotr)  # install the zero on prefix elements
        Result_Matrix[index,] = Vecotr  # rebuild the Result Matrix
        temp.append(0)  # added the zero
    return (Result_Matrix)


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
def GaussJordan(A, b):
    AllMatrix = np.concatenate((A, b), axis=1) * 1.0  # force to float format in Python
    i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
    SubMatrix = AllMatrix
    for ii in range(0, i):
        SubMatrix = GaussJordan_approach(ii, SubMatrix)
    AllMatrix = np.matrix(np.eye(i) * 1)
    AllMatrix = np.append(AllMatrix, SubMatrix, axis=1)
    return (AllMatrix)


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
def Matrix_Inverse(A, I):
    i, j = I.shape[0], I.shape[1]  # parse the matrix dimension information
    AllMatrix = GaussJordan(A, I)
    AllMatrix = AllMatrix[0:i, i:2 * i]
    return (AllMatrix)


def rref_solver(A, b):
    AllMatrix = np.concatenate((A, b), axis=1) * 1.0  # force to float format in Python
    i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
    free_variable=np.ones((j-1,1))
    try:
        AllMatrix = GaussJordan(A, b)
        root_vector = AllMatrix[0:i, j - 1:j]
        if (len(root_vector) == (j - 1)):
            status = 1  # unique solution
        else:
            for ff in range (0,len(root_vector)):
                free_variable[ff,0]=root_vector[ff]-1
            root_vector=np.vstack([root_vector,0])
            status = 2  # multi solutions
    except UnboundLocalError:
        root_vector = 0
        status = 3  # no solution
    return (status, root_vector,free_variable)


import numpy as np


def main():
    print
    "let's make it happen!"
    A1 = np.matrix([
        [3, 1, 1, 1],
        [1, 2, 4, 8],
        [9, 3, 9, 27],
        [-3, 4, 16, 64]
    ])
    b1 = np.matrix([
        [1],
        [9],
        [16],
        [8]
    ])

    A2 = np.matrix([
        [1, 1, 1, 1],
        [1, 2, 4, 8],
        [1, 3, 9, 27],
        [1, 4, 16, 64]
    ])
    b2 = np.matrix([
        [1],
        [10],
        [35],
        [84]
    ])

    A3 = np.random.random((5, 5))
    b3 = np.random.random((5, 1))

    A4 = np.random.random((5, 5)) - A3 * 1j
    b4 = np.random.random((5, 1)) + b3 * 1j

    A5 = np.random.random((5, 5)) - A4 * A3
    b5 = np.random.random((5, 1)) - b4 + b3

    print("\n\nProblem 1-1 for Gauss Elimination")
    Gauss_Matrix = Gauss_eimination(A1, b1).round(3)
    print("\nMatrix A|b 1\n", np.concatenate((A1, b1), axis=1).round(3) * 1.0, "\nGauss Matrix 1\n", Gauss_Matrix)

    Gauss_Matrix = Gauss_eimination(A2, b2).round(3)
    print("\n\nMatrix A|b 2\n", np.concatenate((A2, b2), axis=1).round(3) * 1.0, "\nGauss Matrix 2\n", Gauss_Matrix)

    Gauss_Matrix = Gauss_eimination(A3, b3).round(3)
    print("\n\nMatrix A|b 3\n", np.concatenate((A3, b3), axis=1).round(3) * 1.0, "\nGauss Matrix 3\n", Gauss_Matrix)

    Gauss_Matrix = Gauss_eimination(A4, b4).round(3)
    print("\n\nMatrix A|b 4\n", np.concatenate((A4, b4), axis=1).round(3) * 1.0, "\nGauss Matrix 4\n", Gauss_Matrix)

    Gauss_Matrix = Gauss_eimination(A5, b5).round(3)
    print("\n\nMatrix A|b 5\n", np.concatenate((A5, b5), axis=1).round(3) * 1.0, "\nGauss Matrix 5\n", Gauss_Matrix)

    # Problem 1-2 for Inverse
    print("\n\nProblem 1-2 for Gauss-Jordan")
    Gauss_Matrix = GaussJordan(A1, b1).round(3)
    print("\nMatrix A|b 1\n", np.concatenate((A1, b1), axis=1).round(3) * 1.0, "\nGauss-Jordan Matrix 1\n",
          Gauss_Matrix)

    Gauss_Matrix = GaussJordan(A2, b2).round(3)
    print("\n\nMatrix A|b 2\n", np.concatenate((A2, b2), axis=1).round(3) * 1.0, "\nGauss-Jordan Matrix 2\n",
          Gauss_Matrix)

    Gauss_Matrix = GaussJordan(A3, b3).round(3)
    print("\n\nMatrix A|b 3\n", np.concatenate((A3, b3), axis=1).round(3) * 1.0, "\nGauss-Jordan Matrix 3\n",
          Gauss_Matrix)

    Gauss_Matrix = GaussJordan(A4, b4).round(3)
    print("\n\nMatrix A|b 4\n", np.concatenate((A4, b4), axis=1).round(3) * 1.0, "\nGauss-Jordan Matrix 4\n",
          Gauss_Matrix)

    Gauss_Matrix = GaussJordan(A5, b5).round(3)
    print("\n\nMatrix A|b 5\n", np.concatenate((A5, b5), axis=1).round(3) * 1.0, "\nGauss-Jordan Matrix 5\n",
          Gauss_Matrix)

    # Problem 2
    print("\n\nProblem 2 for Inverse Matrix")
    Ib1 = np.matrix(np.eye(A1.shape[0]))
    Gauss_Matrix = Matrix_Inverse(A1, Ib1)
    print("\nMatrix A|b 1\n", np.concatenate((A1, Ib1), axis=1).round(3) * 1.0, "\nInversed Matrix 1\n",
          Gauss_Matrix.round(3))

    Ib2 = np.matrix(np.eye(A2.shape[0]))
    Gauss_Matrix = Matrix_Inverse(A2, Ib2)
    print("\n\nMatrix A|b 2\n", np.concatenate((A2, Ib2), axis=1).round(3) * 1.0, "\nInversed Matrix 2\n",
          Gauss_Matrix.round(3))

    Ib3 = np.matrix(np.eye(A3.shape[0]))
    Gauss_Matrix = Matrix_Inverse(A3, Ib3)
    print("\n\nMatrix A|b 3\n", np.concatenate((A3, Ib3), axis=1).round(3) * 1.0, "\nInversed Matrix 3\n",
          Gauss_Matrix.round(3))

    Ib4 = np.matrix(np.eye(A4.shape[0]))
    Gauss_Matrix = Matrix_Inverse(A4, Ib4)
    print("\n\nMatrix A|b 4\n", np.concatenate((A4, Ib4), axis=1).round(3) * 1.0, "\nInversed Matrix 4\n",
          Gauss_Matrix.round(3))

    Ib5 = np.matrix(np.eye(A5.shape[0]))
    Gauss_Matrix = Matrix_Inverse(A5, Ib5)
    print("\n\nMatrix A|b 5\n", np.concatenate((A5, Ib5), axis=1).round(3) * 1.0, "\nInversed Matrix 5\n",
          Gauss_Matrix.round(3))

    # Problem 3
    print("\n\nProblem 3 for Equation solver")
    msg = ["reserved", "unique solution", "multi solution", "no solution"]

    status, solution_vector,free_variable = rref_solver(A2, b2)
    print("\nSolution 1: ", msg[status], "\nRREF function result :\n", solution_vector.round(3))
    if (status!=2):
        pass
    else:
        print("remain vector :\n",free_variable)

    AA = np.matrix([
        [1, 1],
        [2, 2]
    ])
    bb = np.matrix([
        [3],
        [5]
    ])
    status, solution_vector,free_variable = rref_solver(AA, bb)
    print("\nSolution 2: ", msg[status], "\nRREF function result :\n", solution_vector)
    if (status!=2):
        pass
    else:
        print("remain vector :\n",free_variable)



    AA3 = np.matrix([
        [1, 1, 1],
        [2, -1, 1]
    ])
    bb3 = np.matrix([
        [1],
        [0]
    ])
    status, solution_vector,free_variable = rref_solver(AA3, bb3)
    print("\nSolution 3: ", msg[status], "\nRREF function result :\n", solution_vector)
    if (status!=2):
        pass
    else:
        print("remain vector :\n",free_variable)

    status, solution_vector,free_variable = rref_solver(A4, b4)
    print("\nSolution 4: ", msg[status], "\nRREF function result :\n", solution_vector)
    if (status!=2):
        pass
    else:
        print("remain vector :\n",free_variable)

    status, solution_vector,free_variable = rref_solver(A5, b5)
    print("\nSolution 5: ", msg[status], "\nRREF function result :\n", solution_vector)
    if (status!=2):
        pass
    else:
        print("remain vector :\n",free_variable)


if __name__ == "__main__":
    main()
