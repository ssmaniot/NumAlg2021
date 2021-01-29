import numpy as np

def Jacobi(A, x0, b, tol = 1.e-10, max_iter = 200):
	n = A.shape[0]
	xk = x0
	k = 0
	while np.linalg.norm(A @ xk - b) > tol and k < max_iter:
		xkk = np.empty(n)
		for i in range(n):
			sigma = 0
			for j in range(n):
				if j != i:
					sigma = sigma + A[i,j] * xk[j]
			xkk[i] = 1/A[i,i] * (b[i] - sigma)
		xk = xkk 
		k += 1
	return xk 

def GaussSeidel(A, x0, b, tol = 1.e-10, max_iter = 200):
	n = A.shape[0]
	xk = x0 
	k = 0
	while np.linalg.norm(A @ xk - b) > tol and k < max_iter:
		xkk = np.empty(n)
		for i in range(n):
			sigma = 0
			for j in range(n):
				if j != i:
					sigma = sigma + A[i,j] * xk[j]
			xkk[i] = (b[i] - sigma) / A[i,i]
		xk = xkk 
		k += 1
	return xk 

def SteepestDescent(A, x0, b, tol = 1.e-10, max_iter = 200):
	r = b - A @ x0
	x = x0 
	k = 0
	while np.linalg.norm(r) > tol and k < max_iter:
		alpha = (r @ r) / (r @ A @ r)
		x = x + alpha * r 
		r = r - alpha * A @ r 
		k += 1
	return x 

def ConjugateGradient(A, x0, b, tol = 1.e-10, max_iter = 200):
	r = b - A @ x0
	p = r 
	x = x0 
	k = 0
	while np.linalg.norm(r) > tol and k < max_iter:
		Ap = A @ p 
		pAp = p @ Ap 
		alpha = (p @ r) / pAp 
		x = x + alpha * p 
		r = r - alpha * Ap 
		beta = (r @ Ap) / pAp 
		p = r + beta * p 
		k += 1
	return x 

def PreconditionedConjugateGradient(A, x0, b, tol = 1.e-10, max_iter = 200):
	P = np.diag(np.diag(A))
	r = b - A @ x0 
	p = r 
	x = x0 
	k = 0
	while np.linalg.norm(r) / np.linalg.norm(b) > tol and k < max_iter:
		Ap = A @ p 
		pAp = p @ Ap
		alpha = (p @ r) / pAp 
		x = x + alpha * p 
		r = r - alpha * Ap
		z = np.linalg.solve(P, r)
		beta = (z @ Ap) / pAp 
		p = z + beta * p 
		k += 1
	return x 

A = np.array([[3., -2.], [1., 1.]])
b = np.array([1., 1.])
x0 = np.array([-1., 3.])
x = PreconditionedConjugateGradient(A, x0, b)
print(A @ x - b)