#   # -*- coding: UTF-8 -*-
#   trial on the : Satomi machine
#   Created by Ush on 2018/3/30-04/13 for the LU decomposition
#   Project name :  Class06
#   Please contact CHIH, HSIN-CHING/D0631008 when expect to refer this source code.
#   NOTE : no liability on any loss nor damage by using this source code. it is your own risk.

from __future__ import division


# https://blog.csdn.net/u013088062/article/details/50353202
# https://stackoverflow.com/questions/37515461/sorting-all-rows-in-numpy-matrix-by-target-column
def lu_colx(Matrix):
    AllMatrix = Matrix * 1.  # force to float format in Python
    i, j = AllMatrix.shape[0], AllMatrix.shape[1]  # parse the matrix dimension information
    U_Matrix = AllMatrix * 0.  # create the result matrix with all zero with float format
    L_Matrix = np.matrix(np.eye(j)) * 1.  # create the L_matrix and force in the float
    P_Matrix = np.matrix(np.eye(j)) * 1.
    SubMatrix = AllMatrix
    SubMatrix, P_original = Matrix_sort(SubMatrix)  # derive the P_orignal vector
    for index in range(0, i):
        SubMatrix, Vector, P, L_vector = Row_handle(SubMatrix)
        U_Matrix[index, index:j] = Vector
        L_Matrix[index:i, index] = L_vector + L_Matrix[index:i, index]
        if (L_Matrix[index + 1:i, 0:index + 1].shape[0] > 1.99):  # start the P-checking
            L_Matrix[index + 1:i, 0:index + 1] = L_Matrix_handle(L_Matrix[index + 1:i, 0:index + 1], P)
        if (index < i - 2):  # stop P matrix management in the too small case for P.shape issue
            P_original = P_handle(P_original, P)
    P_Matrix = P_parse(np.matrix(np.eye(j)) * 0., P_original)
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
        SubMatrix, P = Matrix_sort(SubMatrix)  # lift up the bigger pivot and report P
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


import scipy.linalg as la


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
    L, U, P = lu_colx(A)
    Pb = np.dot(P, b)
    Z = forward(L, Pb)
    X = backsubs(U, Z)
    return (X)


# http://www.iiserpune.ac.in/~pgoel/LUDecomposition.pdf
def forward(L, Pb):
    # forward elimination L*Z=Pb ,
    # input : L , Pb
    # output : Z
    i, j = L.shape[0], L.shape[1]
    Z = Pb * 0.  # create the Z report vector and force to float
    Z[0] = Pb[0, 0]
    for index_i in range(0, i):  # 1 to i
        Z[index_i] = Pb[index_i] - np.dot(L[index_i, 0:index_i], Z[0:index_i, ])
    return Z


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


def Matrix_inverse(A):
    # input : Matrix A
    # output : X
    # X should be the inversed matrix of input Matrix A
    # expectation : Call lu_colx(A) => LZ=I => UX^=Z => X=P*X^
    L, U, P = lu_colx(A)  # AX=I => LUX=P*I => LUX=P
    Z = forward(L, P)  # LUX=P => LZ=P => find Z
    X = backsubs(U, Z)  # UX=Z => find X equal A^-1
    return (X)


# https://keisan.casio.com/exec/system/15076953047019
#
def main():
    print
    "let's make it happen!"

    A1 = np.matrix([
        [2, 1, 1],
        [1.99, 2, -1],
        [4, -1, 6]

    ])
    b1 = np.matrix([
        [1],
        [2],
        [3]
    ])

    A2 = np.matrix([
        [1, 2, 3],
        [4, 5, 8],
        [10, 12, 3]
    ])

    b2 = np.matrix([
        [1],
        [10],
        [84]
    ])

    A3 = np.matrix([
        [8, 1, 6],
        [3, 5, 7],
        [4, 9, 2]

    ])
    b3 = np.matrix([
        [9],
        [6],
        [-1]
    ])

    A4 = np.matrix([
        [1, 0.533, -0.6543, 1],
        [2, 9.2, 1.654, 1.2],
        [19, 2, 1, -2],
        [2, 0.07, 6, 1.097]
    ])
    b4 = np.matrix([
        [2],
        [1],
        [2],
        [1]
    ])

    A5 = np.matrix([
        [1, 0.5, -0.6, 1, 0.6],
        [2, 9.2, 1.6, 1.2, -0.5],
        [3, 1.2, 1, -2, 0.6],
        [1.5, 2, 1, -2, 1.2],
        [2, 0.7, 6, 1.97, 5]
    ])
    b5 = np.matrix([
        [2],
        [1],
        [3.3],
        [2.1],
        [1]
    ])

    print("\n\nProblem 1-1 for LU decomposition")
    print("\n\nInput_Matrix\n", A1)
    L, U, Q = lu_colx(A1)
    print("Result_Matrix-A1 U\n", U)
    print("Result_Matrix-A1 Q\n", Q)
    print("Result_Matrix-A1 L\n", L)
    print("\nError1 -A1 Norm(AQ-LU): ", la.norm(np.dot(L, U) - np.dot(Q, A1)))
    X1 = Axb_solver(A1, b1)
    print("Error2 -A1 Norm(b-Ax): ", la.norm(b1 - np.dot(A1, X1)))
    X1D = Matrix_inverse(A1)
    print("Error3 -A1 Norm(I-AX): ", la.norm((np.matrix(np.eye(A1.shape[0]))) - np.dot(A1, X1D)))

    # http://libai.math.ncu.edu.tw/webclass/matrix/ch1_7/
    print("\n\nInput_Matrix A2\n", A2)
    L, U, Q = lu_colx(A2)
    print("Result_Matrix-A2 U\n", U)
    print("Result_Matrix-A2 Q\n", Q)
    print("Result_Matrix-A2 L\n", L)
    print("\nError1 -A2 Norm(AQ-LU): ", la.norm(np.dot(L, U) - np.dot(Q, A2)))
    X2 = Axb_solver(A2, b2)
    print("Error2 -A2 Norm(b-Ax): ", la.norm(b2 - np.dot(A2, X2)))
    X2 = Matrix_inverse(A2)
    print("Error3 -A2 Norm(I-AX): ", la.norm((np.matrix(np.eye(A2.shape[0]))) - A2 * X2))

    # https://math.stackexchange.com/questions/1009916/easy-way-to-calculate-inverse-of-an-lu-decomposition
    print("\n\nInput_Matrix A3\n", A3)
    L, U, Q = lu_colx(A3)
    print("Result_Matrix-A3 U\n", U)
    print("Result_Matrix-A3 Q\n", Q)
    print("Result_Matrix-A3 L\n", L)
    print("\nError1 -A3 Norm(AQ-LU): ", la.norm(np.dot(L, U) - np.dot(Q, A3)))
    X3 = Axb_solver(A3, b3)
    print("Error2 -A3 Norm(b-Ax): ", la.norm(b3 - np.dot(A3, X3)))
    X3 = Matrix_inverse(A3)
    print("Error3 -A3 Norm(I-AX): ", la.norm((np.matrix(np.eye(A3.shape[0]))) - A3 * X3))

    print("\n\nInput_Matrix A4\n", A4)
    L, U, Q = lu_colx(A4)
    print("Result_Matrix-A4 U\n", U)
    print("Result_Matrix-A4 P\n", Q)
    print("Result_Matrix-A4 L\n", L)
    print("\nError1 -A4 Norm(AQ-LU): ", la.norm(np.dot(L, U) - np.dot(Q, A4)))
    X4 = Axb_solver(A4, b4)
    print("Error2 -A4 Norm(b-Ax): ", la.norm(b4 - np.dot(A4, X4)))
    X4 = Matrix_inverse(A4)
    print("Error3 -A4 Norm(I-AX): ", la.norm((np.matrix(np.eye(A4.shape[0]))) - A4 * X4))

    print("\n\nInput_Matrix A5\n", A5)
    L, U, Q = lu_colx(A5)
    print("Result_Matrix-A5 U\n", U)
    print("Result_Matrix-A5 P\n", Q)
    print("Result_Matrix-A5 L\n", L)
    print("\nError1 -A5 Norm(AQ-LU): ", la.norm(np.dot(L, U) - np.dot(Q, A5)))
    X5 = Axb_solver(A5, b5)
    print("Error2 -A5 Norm(b-Ax): ", la.norm(b5 - np.dot(A5, X5)))
    X5 = Matrix_inverse(A5)
    print("Error3 -A5 Norm(I-AX): ", la.norm((np.matrix(np.eye(A5.shape[0]))) - A5 * X5))


if __name__ == "__main__":
    main()
