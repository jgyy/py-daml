"""
SciPy
"""
from numpy.random import rand
from numpy import array, dot
from scipy import linalg, sparse

# Linear Algebra
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])

# Compute the determinant of a matrix
print(linalg.det(A))
P, L, U = linalg.lu(A)
print(P)
print(L)
print(U)
print(dot(L, U))
EW, EV = linalg.eig(A)
print(EW)
print(EV)
v = array([[2], [3], [5]])
print(v)
s = linalg.solve(A, v)
print(s)

# Row-based linked list sparse matrix
A = sparse.lil_matrix((1000, 1000))
print(A)
A[0,:100] = rand(100)
A[1,100:200] = A[0,:100]
A.setdiag(rand(1000))
print(A)
A = A.tocsr()
b = rand(1000)
print(linalg.spsolve(A, b))
