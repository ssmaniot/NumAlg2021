import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, spsolve_triangular, eigs
from scipy.io import loadmat
from time import process_time
import matplotlib.pyplot as plt
	
def Jacobi(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = process_time()
	n = A.shape[0]
	x = x0
	d = 1. / A.diagonal()
	DA = A.copy()
	if isinstance(DA, np.ndarray):
		np.fill_diagonal(DA, 0.)
	else:
		DA.setdiag(0.)
	bnorm = np.linalg.norm(b)
	res = [np.linalg.norm(A.dot(x) - b) / bnorm]
	k = 0
	while res[-1] > tol and k < max_iter:
		x = d * (b - DA.dot(x))
		res.append(np.linalg.norm(A.dot(x) - b) / bnorm)
		k += 1
	return x, k, np.array(res), process_time() - start
	
def GaussSeidel(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = process_time()
	n = A.shape[0]
	x = x0 
	if isinstance(A, np.ndarray):
		MD, N = np.tril(A), np.triu(A, k = 1)
		linear_solver = np.linalg.solve
	else:
		MD, N = sps.tril(A, format = 'csr'), sps.triu(A, k = 1, format = 'csr')
		linear_solver = spsolve_triangular
	bnorm = np.linalg.norm(b)
	res = [np.linalg.norm(A.dot(x) - b) / bnorm]
	k = 0
	while res[-1] > tol and k < max_iter:
		x = linear_solver(MD, b - N.dot(x)) 
		res.append(np.linalg.norm(A.dot(x) - b) / bnorm)
		k += 1 
	return x, k, np.array(res), process_time() - start

def GaussSeidelCC(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = process_time()
	n = A.shape[0]
	x = x0 
	d = A.diagonal()
	if isinstance(A, np.ndarray):
		M, N = np.tril(A, k = -1), np.triu(A, k = 1)
	else:
		M, N = sps.tril(A, k = -1, format = 'csr'), sps.triu(A, k = 1, format = 'csr')
	bnorm = np.linalg.norm(b)
	res = [np.linalg.norm(A.dot(x) - b) / bnorm]
	k = 0
	while res[-1] > tol and k < max_iter:
		v = N.dot(x)
		xk = np.zeros(n)
		for i in range(n):
			s = M[i,:].dot(xk)
			xk[i] = (b[i] - v[i] - s) / d[i]
		x = xk
		res.append(np.linalg.norm(A.dot(x) - b) / bnorm)
		k += 1
	return x, k, np.array(res), process_time() - start

def SteepestDescent(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = process_time()
	r = b - A.dot(x0)
	x = x0 
	bnorm = np.linalg.norm(b)
	res = [np.linalg.norm(r) / bnorm]
	k = 0
	while res[-1] > tol and k < max_iter:
		Ar = A.dot(r)
		alpha = (r @ r) / (r @ Ar)
		x = x + alpha * r 
		r = r - alpha * Ar 
		res.append(np.linalg.norm(r) / bnorm)
		k += 1
	return x, k, np.array(res), process_time() - start

def ConjugateGradient(A, x0, b, tol = 1.e-10, max_iter = 200):
	start = process_time()
	r = b - A.dot(x0)
	p = r 
	x = x0 
	bnorm = np.linalg.norm(b)
	res = [np.linalg.norm(r) / bnorm]
	k = 0
	while res[-1] > tol and k < max_iter:
		Ap = A.dot(p) 
		pAp = p @ Ap 
		alpha = (p @ r) / pAp 
		x = x + alpha * p 
		r = r - alpha * Ap 
		beta = (r @ Ap) / pAp 
		p = r + beta * p 
		res.append(np.linalg.norm(r) / bnorm)
		k += 1
	return x, k, np.array(res), process_time() - start

def PreconditionedConjugateGradient(A, x0, b, P, tol = 1.e-10, max_iter = 200):
	start = process_time()
	n = A.shape[0]
	r = b - A.dot(x0) 
	p = r 
	x = x0 
	bnorm = np.linalg.norm(b)
	res = [np.linalg.norm(r) / bnorm]
	k = 0
	while res[-1] > tol and k < max_iter:
		Ap = A.dot(p)
		pAp = p @ Ap
		alpha = (p @ r) / pAp 
		x = x + alpha * p 
		r = r - alpha * Ap
		z = spsolve(P, r)
		beta = (z @ Ap) / pAp 
		p = z + beta * p 
		res.append(np.linalg.norm(r) / bnorm)
		k += 1
	return x, k, np.array(res), process_time() - start

if __name__ == '__main__':

	A = sps.load_npz('A.npz')
	P = sps.load_npz('P.npz')
	n = A.shape[0]
	b = np.ones(n)
	np.random.seed(1)
	x0 = np.random.rand(n)
	
	# Part 1: Relative Residuals Diagram 

	conv_diag = plt.figure()

	x, k, r, t = Jacobi(A, x0, b)
	plt.plot(np.arange(k + 1), r, ':o', label = 'Jacobi')
	print('iter', k)
	print('time', t)
	print('J   ', np.linalg.norm(A.dot(x) - b))

	x, k, r, t = GaussSeidel(A, x0, b)
	plt.plot(np.arange(k + 1), r, ':o', label = 'Gauss-Seidel')
	print('iter', k)
	print('time', t)
	print('GS  ', np.linalg.norm(A.dot(x) - b))

	x, k, r, t = GaussSeidelCC(A, x0, b)
	plt.plot(np.arange(k + 1), r, ':o', label = 'Gauss-Seidel (cc)')
	print('iter', k)
	print('time', t)
	print('GScc', np.linalg.norm(A.dot(x) - b))

	x, k, r, t = SteepestDescent(A, x0, b)
	plt.plot(np.arange(k + 1), r, ':o', label = 'Steepest Descent')
	print('iter', k)
	print('time', t)
	print('SD  ', np.linalg.norm(A.dot(x) - b))

	x, k, r, t = ConjugateGradient(A, x0, b)
	plt.plot(np.arange(k + 1), r, ':o', label = 'Conjugate Gradient')
	print('iter', k)
	print('time', t)
	print('CG  ', np.linalg.norm(A.dot(x) - b))

	x, k, r, t = PreconditionedConjugateGradient(A, x0, b, P)
	plt.plot(np.arange(k + 1), r, ':o', label = 'PCG - Incomplete Cholesky')
	print('iter', k)
	print('time', t)
	print('PCGc', np.linalg.norm(A.dot(x) - b))

	D = sps.diags(A.diagonal(), format = 'csr')
	x, k, r, t = PreconditionedConjugateGradient(A, x0, b, D)
	plt.plot(np.arange(k + 1), r, ':o', label = 'PCG - Jacobi')
	print('iter', k)
	print('time', t)
	print('PCGd', np.linalg.norm(A.dot(x) - b))

	plt.title('Convergence Diagram of the Residuals')
	plt.xlabel(r'$k$')
	plt.ylabel(r'$|r^{(k)}|/|b|$')
	plt.yscale('log')
	plt.legend()
		
	plt.savefig('conv_diag_res.pdf')
	plt.clf()
	
	# Part 2/3: Average Time Elapsed 

	def avg_time_elapsed(method, iterations, *argv):
		elapsed_time = np.empty(iterations)
		for i in range(iterations):
			_, _, _, t = method(*argv)
			elapsed_time[i] = t
		return np.mean(elapsed_time), np.std(elapsed_time)

	def standard_solver(A, x0, b):
		linear_solver = np.linalg.solve if isinstance(A, np.ndarray) else spsolve
		start = process_time()
		x = linear_solver(A, b) 
		return x, None, None, process_time() - start

	print('\nAvg Time Elapsed\n')

	# Part 2: Sparse Matrix
	print('Sparse')
	m, sd = avg_time_elapsed(Jacobi, 10, A, x0, b)
	print('J    {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(GaussSeidel, 10, A, x0, b)
	print('GS   {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(GaussSeidelCC, 10, A, x0, b)
	print('GScc {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(SteepestDescent, 10, A, x0, b)
	print('SD   {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(ConjugateGradient, 10, A, x0, b)
	print('CG   {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, P)
	print('PCGc {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, D)
	print('PCGd {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(standard_solver, 10, A, x0, b)
	print('SS   {:.2e} ± {:.2e}'.format(m, sd))
	#hi
	# Part 3: Dense Matrix
	print('\nDense')
	A = A.todense().A 
	m, sd = avg_time_elapsed(Jacobi, 10, A, x0, b)
	print('J    {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(GaussSeidel, 10, A, x0, b)
	print('GS   {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(GaussSeidelCC, 10, A, x0, b)
	print('GScc {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(SteepestDescent, 10, A, x0, b)
	print('SD   {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(ConjugateGradient, 10, A, x0, b)
	print('CG   {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, P)
	print('PCGc {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(PreconditionedConjugateGradient, 10, A, x0, b, D)
	print('PCGd {:.2e} ± {:.2e}'.format(m, sd))
	m, sd = avg_time_elapsed(standard_solver, 10, A, x0, b)
	print('SS   {:.2e} ± {:.2e}'.format(m, sd))