import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.io import loadmat
from time import process_time, time 
	
def Jacobi(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = time()
	n = A.shape[0]
	x = x0
	d = 1 / A.diagonal()
	DA = A.copy()
	DA.setdiag(0.)
	bnorm = np.linalg.norm(b)
	k = 0
	while np.linalg.norm(A.dot(x) - b) / bnorm > tol and k < max_iter:
		x = d * (b - DA.dot(x))
		k += 1
	return x, k, time() - start
	
def GaussSeidel(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = time()
	n = A.shape[0]
	x = x0 
	MD, N = sps.tril(A, format = 'csr'), sps.triu(A, k = 1, format = 'csr')
	bnorm = np.linalg.norm(b)
	k = 0
	while np.linalg.norm(A.dot(x) - b) / bnorm > tol and k < max_iter:
		x = spsolve_triangular(MD, b - N.dot(x)) 
		k += 1 
	return x, k, time() - start

def GaussSeidelCC(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = time()
	n = A.shape[0]
	x = x0 
	d = A.diagonal()
	M, N = sps.tril(A, k = -1, format = 'csr'), sps.triu(A, k = 1, format = 'csr')
	bnorm = np.linalg.norm(b)
	k = 0
	while np.linalg.norm(A.dot(x) - b) / bnorm > tol and k < max_iter:
		v = N.dot(x)
		xk = np.zeros(n)
		for i in range(n):
			s = M[i,:].dot(xk)
			xk[i] = (b[i] - v[i] - s) / d[i]
		x = xk
		k += 1
	return x, k, time() - start

def SteepestDescent(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = time()
	r = b - A.dot(x0)
	x = x0 
	bnorm = np.linalg.norm(b)
	k = 0
	while np.linalg.norm(r) / bnorm > tol and k < max_iter:
		Ar = A.dot(r)
		alpha = (r @ r) / (r @ Ar)
		x = x + alpha * r 
		r = r - alpha * Ar 
		k += 1
	return x, k, time() - start

def ConjugateGradient(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = time()
	r = b - A.dot(x0)
	p = r 
	x = x0 
	bnorm = np.linalg.norm(b)
	k = 0
	while np.linalg.norm(r) / bnorm > tol and k < max_iter:
		Ap = A.dot(p) 
		pAp = p @ Ap 
		alpha = (p @ r) / pAp 
		x = x + alpha * p 
		r = r - alpha * Ap 
		beta = (r @ Ap) / pAp 
		p = r + beta * p 
		k += 1
	return x, k, time() - start

def PreconditionedConjugateGradient(A, x0, b, P, tol = 1.e-10, max_iter = 200):
	start = time()
	n = A.shape[0]
	r = b - A.dot(x0) 
	p = r 
	x = x0 
	bnorm = np.linalg.norm(b)
	k = 0
	while np.linalg.norm(r) / bnorm > tol and k < max_iter:
		Ap = A.dot(p)
		pAp = p @ Ap
		alpha = (p @ r) / pAp 
		x = x + alpha * p 
		r = r - alpha * Ap
		z = spsolve(P, r)
		beta = (z @ Ap) / pAp 
		p = z + beta * p 
		k += 1
	return x, k, time() - start

i = loadmat('savei.mat')['i'][:,0] - 1
j = loadmat('savej.mat')['j'][:,0] - 1
n = max(np.max(i), np.max(j)) + 1
print('Matrix of size', n)
A = sps.csr_matrix((-np.ones(i.shape[0]), (i, j)), (n, n), dtype = np.float)
A = A + A.T 
d = np.abs(A.sum()) + 1
A = A + sps.diags(d * np.ones(n))
b = np.ones(n)
np.random.seed(1)
x0 = np.random.rand(n)

x, k, t = Jacobi(A, x0, b)
print('iter', k)
print('time', t)
print('J   ', np.linalg.norm(A.dot(x) - b))
x, k, t = GaussSeidel(A, x0, b)
print('iter', k)
print('time', t)
print('GS  ', np.linalg.norm(A.dot(x) - b))
x, k, t = GaussSeidelCC(A, x0, b)
print('iter', k)
print('time', t)
print('GScc', np.linalg.norm(A.dot(x) - b))
x, k, t = SteepestDescent(A, x0, b)
print('iter', k)
print('time', t)
print('SD  ', np.linalg.norm(A.dot(x) - b))
x, k, t = ConjugateGradient(A, x0, b)
print('iter', k)
print('time', t)
print('CG  ', np.linalg.norm(A.dot(x) - b))
P = sps.csr_matrix((A.diagonal(), (np.arange(n), np.arange(n))))
x, k, t = PreconditionedConjugateGradient(A, x0, b, P)
print('iter', k)
print('time', t)
print('PCG ', np.linalg.norm(A.dot(x) - b))

# n = 10000000
# data = np.random.rand(n//2)
# A = sps.csr_matrix((data, (np.random.randint(0, n, n//2), np.random.randint(0, n, n//2))), (n, n), dtype = np.float)
# D = sps.eye(n, dtype = np.float)
# A = 0.5 * (A + A.T) + n * D 
# b = np.ones(n)
# x0 = np.random.rand(n) * np.array(np.random.rand(n) < 0.5) 