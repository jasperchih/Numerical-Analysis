#   # -*- coding: UTF-8 -*-
#   trial on the : Satomi machine
#   Created by Ush on 2018/3/30-04/13 for the LU decomposition
#   Project name :  Class06
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


def main():
    print
    "let's make it happen!"

    # type = NCM07.definite(A2)

import time
if __name__ == "__main__":
    main()
    A1 = np.matrix([
        [1, 2, 3],
        [4, 5, 8],
        [10, 12, 3]
    ])
    b1 = np.matrix([
        [1],
        [10],
        [84]
    ])

    B0 = np.matrix([
        [6.857, -4, -2.857, 0],
        [-4, 74, -3.333, -66.666],
        [-2.857, -3.333, 39.523, 0],
        [0, -66.666, 0, 66.666]

    ])
    P = np.matrix([
        [-1.6],
        [-2],
        [-3.7],
        [5]
    ])


    B1 = np.matrix([
        [-30, 0, 10, 20],
        [0, -100, 100, 0],
        [10, 100, -150, 40],
        [20, 0, 40, -110]

    ])
    P1 = np.matrix([
        [-8],
        [4.4],
        [0],
        [0]
    ])

    AA=A1
    bb=b1
    t0=time.time()
    status, root_vector, free_variable = NCM05.rref_solver(AA, bb)
    t1 = time.time()
    x1 = NCM07.choleski(AA, bb)
    t2 = time.time()
    x2 = NCM06.Axb_solver(AA, bb)
    t3 = time.time()
    x3 = NCM08.gram_solver(AA, bb)
    t4 = time.time()
    print("\n\n RREF solver\n", root_vector)
    print(" Choleski solver\n", x1)
    print(" LU Decomposition solver\n", x2)
    print(" Gram solver\n", x3)
    print("RREF solver time ",t1-t0)
    print("Choleski solver time ", t2 - t1)
    print("LU Decomposition time ", t3 - t2)
    print("Gram solver time ", t4 - t3)


    AA=B0
    bb=P
    t0=time.time()
    status, root_vector, free_variable = NCM05.rref_solver(AA, bb)
    t1 = time.time()
    x1 = NCM07.choleski(AA, bb)
    t2 = time.time()
    x2 = NCM06.Axb_solver(AA, bb)
    t3 = time.time()
    x3 = NCM08.gram_solver(AA, bb)
    t4 = time.time()
    print("\n\n RREF solver\n", root_vector)
    print(" Choleski solver\n", x1)
    print(" LU Decomposition solver\n", x2)
    print(" Gram solver\n", x3)
    print("RREF solver time ",t1-t0)
    print("Choleski solver time ", t2 - t1)
    print("LU Decomposition time ", t3 - t2)
    print("Gram solver time ", t4 - t3)


    AA=B1
    bb=P1
    t0=time.time()
    status, root_vector, free_variable = NCM05.rref_solver(AA, bb)
    t1 = time.time()
    x1 = NCM07.choleski(AA, bb)
    t2 = time.time()
    x2 = NCM06.Axb_solver(AA, bb)
    t3 = time.time()
    x3 = NCM08.gram_solver(AA, bb)
    t4 = time.time()
    print("\n\n RREF solver\n", root_vector)
    print(" Choleski solver\n", x1)
    print(" LU Decomposition solver\n", x2)
    print(" Gram solver\n", x3)
    print("RREF solver time ",t1-t0)
    print("Choleski solver time ", t2 - t1)
    print("LU Decomposition time ", t3 - t2)
    print("Gram solver time ", t4 - t3)




