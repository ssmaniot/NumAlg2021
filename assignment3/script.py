import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
from time import process_time
import pandas as pd 

#
# In this assignment, solve (CP)
#
# y'(t) = f(y, y(t)),   t \in [t_0, T]
# y(0) = y_0
#
# with y(t) a vector of equations in t.
#

# Forward Euler solver for systems of 1st order ODE
def ForwardEuler(f, y0, N, T, *argv):
	print('ForwardEuler(N={}, T={})'.format(N, T))
	if type(y0) is not np.ndarray:
		y0 = np.array([y0])
	h = 2. ** (-N)
	time = np.arange(0., T, step = h)
	y = np.empty((time.shape[0], y0.shape[0]))
	y[0] = y0
	for i in range(len(time[:-1])):
		y[i+1] = y[i] + h * f(y[i], time[i], *argv)
	return { 'solution': y, 'time_steps': time.shape[0] }

# Solver for systems of nonlinear equations
def NewtonRaphson(f, Df, p0, max_iter = 1000, tol = 1e-9):
	if type(p0) is not np.ndarray:
		p0 = np.array([p0])
	p = p0
	for k in range(max_iter):
		pk = p - f(p) / Df(p)
		err = np.linalg.norm(pk - p) / np.linalg.norm(pk)
		if err < tol:
			break
		p = pk
	return pk, k + 1
	
# Solver for systems of nonlinear equations
def VariableSecantScheme(f, p0, p1, max_iter = 1000, tol = 1e-9):
	for k in range(max_iter):
		q = (p1 - p0) / (f(p1) - f(p0))
		p0 = p1 
		p1 = p1 - q * f(p1)
		err = np.linalg.norm(p1 - p0) / np.linalg.norm(p1)
		if err < tol:
			break
	return p1, k + 1

# Backward Euler solver for systems of 1st order ODE
def BackwardEuler(f, Df, y0, N, T, *argv):
	print('BackwardEuler(N={}, T={})'.format(N, T))
	if type(y0) is not np.ndarray:
		y0 = np.array([y0])
	h = 2. ** (-N)
	time = np.arange(0., T, step = h)
	y = np.empty((time.shape[0], y0.shape[0]))
	nonlinear_iterations = np.empty(time.shape[0])
	y[0] = y0
	i = 0
	
	def f_(y_):
		return y_ - y[i] - h * f(y_, time[i+1], *argv)
	
	def Df_(y_):
		return 1 - h * Df(y_, time[i+1], *argv)
	
	while i < len(time[:-1]):
		# y[i+1], iter = NewtonRaphson(f_, Df_, y[i])
		y[i+1], iter = VariableSecantScheme(f_, y[i] - 1.e3, y[i] + 1.e3)
		nonlinear_iterations[i] = iter 
		i += 1
	return { 'solution': y, 'time_steps': time.shape[0], 'avg_nonlinear_iter': np.mean(nonlinear_iterations), 'no_nonlinear_convergence': np.sum(nonlinear_iterations == 1000) }

# Crank-Nicolson solver for systems of 1st order ODE
def CrankNicolson(f, Df, y0, N, T, *argv):
	print('CrankNicolson(N={}, T={})'.format(N, T))
	if type(y0) is not np.ndarray:
		y0 = np.array([y0])
	h = 2. ** (-N)
	time = np.arange(0., T, step = h)
	y = np.empty((time.shape[0], y0.shape[0]))
	nonlinear_iterations = np.empty(time.shape[0])
	y[0] = y0 
	i = 0
	
	def f_(y_):
		return y_ - y[i] - 0.5 * h * (f(y_, time[i+1], *argv) + f(y[i], time[i], *argv))
	
	def Df_(y_):
		return 1 - 0.5 * h * (Df(y_, time[i+1], *argv) + Df(y[i], time[i], *argv))
	
	while i < len(time[:-1]):
		# y[i+1], iter = NewtonRaphson(f_, Df_, y[i])
		y[i+1], iter = VariableSecantScheme(f_, y[i] - 1.e3, y[i] + 1.e3)
		nonlinear_iterations[i] = iter 
		i += 1
	return { 'solution': y, 'time_steps': time.shape[0], 'avg_nonlinear_iter': np.mean(nonlinear_iterations), 'no_nonlinear_convergence': np.sum(nonlinear_iterations == 1000) }

# Printer function for display info about method execution
def print_execution_info(data, elapsed_time, implicit):
	print('- Elapsed CPU time:                       {}'.format(elapsed_time))
	print('- # of time steps:                        {}'.format(data['time_steps']))
	if implicit:
		print('- Avg # of nonlinear iter. per time-step: {}'.format(data['avg_nonlinear_iter']))
		print('- # of time steps with no convergence:    {}'.format(data['no_nonlinear_convergence']))
	print('\n')

#######################################################

if __name__ == '__main__':

	"""
	PART 1: SPRING EQUATION
	"""

	# Function of spring model 
	def spring(z, t, k, m):
		y = z[0]
		v = z[1]
		dydt = v 
		dvdt = -k/m * y 
		return np.array([dydt, dvdt])
	
	# First derivative of spring model 
	def dspring(z, t, k, m):
		y = z[0]
		v = z[1]
		d2ydt2 = (-k / m) * y
		d2vdt2 = (-k / m) * v
		return np.array([d2ydt2, d2vdt2])
	
	# Parameters of the CP problem
	z0 = np.array([0.1, 0.])
	k = 1.
	m = 0.25
	T = 15
	N = 8
	
	# Allocating array for error 
	err = np.ones((N, 4))
	
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle('Comparison Between Numerical and Analytical Solutions', fontsize=30)
	
	for n in range(1, N + 1):
		print('---')
		h = 2. ** (-n)
		time = np.arange(0., T, step=h)
		
		# Solving using numerical schemes
		
		# Forward Euler 
		begin = process_time()
		fe = ForwardEuler(spring, z0, n, T, k, m)
		end = process_time()
		print_execution_info(fe, end - begin, implicit = False)
		zfe = fe['solution']
		
		# Backward Euler
		begin = process_time()
		be = BackwardEuler(spring, dspring, z0, n, T, k, m)
		end = process_time()
		print_execution_info(be, end - begin, implicit = True)
		zbe = be['solution']
		
		# Crank-Nicolson 
		begin = process_time()
		cn = CrankNicolson(spring, dspring, z0, n, T, k, m)
		end = process_time()
		print_execution_info(cn, end - begin, implicit = True)
		zcn = cn['solution']
		
		# scipy.integrate.odeint()
		print('scipy.integrate.odeint()')
		begin = process_time()
		zoi = odeint(spring, z0, time, args=(k, m))
		end = process_time()
		print('- Elapsed CPU time:                       {}'.format(end - begin))
		print('\n')
		
		# Analytical solution
		zan = z0[0] * np.cos(np.sqrt(k / m) * time)
		
		# Error computation
		err[n-1,:] = np.min(np.abs(zfe[1:,0] - zan[1:])), np.min(np.abs(zbe[1:,0] - zan[1:])), np.min(np.abs(zcn[1:,0] - zan[1:])), np.min(np.abs(zoi[1:,0] - zan[1:]))
		
		# Plotting results for N=1 and N=8
		if n == 1 or n == 8:
			idx = 0 if n == 1 else 1
			axes[idx].plot(time, zfe[:,0], '-', label = 'Forward Euler')
			axes[idx].plot(time, zbe[:,0], '--', label = 'Backward Euler')
			axes[idx].plot(time, zcn[:,0], '-.', label = 'Crank-Nicolson')
			axes[idx].plot(time, zoi[:,0], ':', label = 'scipy.integrate.odeint()')
			axes[idx].set_xlabel(r'time', fontsize=18)
			axes[idx].set_ylabel(r'$y(t)$', fontsize=18)
			axes[idx].set_title('Spring Equation for $N={}$'.format(n), fontsize=18)
			axes[idx].legend()
	
	plt.savefig('numerical_analytical_comp.pdf')
	plt.clf()
	
	models = ['Forward Euler', 'Backward Euler', 'Crank-Nicolson', 'scipy.integrate.odeint()']
	
	# Plotting the convergence of error and error ratio
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle('Convergence of error', fontsize=30)
	
	axes[0].plot(np.arange(1, N + 1), err, ':o')
	axes[0].set_xlabel(r'$N$', fontsize=18)
	axes[0].set_ylabel(r'$e^{(N)}$', fontsize=18)
	axes[0].set_title(r'Absolute Error', fontsize=18)
	axes[0].legend(models)
	
	axes[1].plot(np.arange(2, N + 1), err[1:] / err[:-1], ':o')
	axes[1].set_xlabel(r'$N$', fontsize=18)
	axes[1].set_ylabel(r'$e^{(N)}/e^{(N-1)}$', fontsize=18)
	axes[1].set_title(r'Error Ratio', fontsize=18)
	axes[1].legend(models)
	
	plt.savefig('error.pdf')
	plt.clf()
	
	"""
	PART 2: EPIDEMIOLOGICAL MODEL (OPTIONAL)
	"""

	# Derivatives of components of SEIARD model 
	def dSdt(beta0, betaA, A, D, I, S, Np):
		return - beta0 * (I + betaA * A) * S / (Np - D)
	
	def dEdt(beta0, betaA, delta, A, D, E, I, S):
		return - dSdt(beta0, betaA, A, D, I, S, Np) - delta * E 
	
	def dIdt(alpha, gammaI, delta, sigma, E, I):
		return sigma * delta * E - gammaI * I - alpha * I 
	
	def dAdt(gammaA, delta, sigma, A, E):
		return (1. - sigma) * delta * E - gammaA * A 
	
	def dRdt(gammaA, gammaI, A, I):
		return gammaI * I + gammaA * A 
	
	def dDdt(alpha, I):
		return alpha * I

	# Function of SEIARD model 
	def seiard(z, t, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np):
		S, E, I, A, R, D = z  
		return np.array([
			dSdt(beta0, betaA, A, D, I, S, Np), 
			dEdt(beta0, betaA, delta, A, D, E, I, S),
			dIdt(alpha, gammaI, delta, sigma, E, I), 
			dAdt(gammaA, delta, sigma, A, E), 
			dRdt(gammaA, gammaI, A, I), 
			dDdt(alpha, I)])
	
	# First derivative of SEIARD model 
	def dseiard(z, t, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np):
		S, E, I, A, R, D = z 
		dsdt = dSdt(beta0, betaA, A, D, I, S, Np)
		dedt = dEdt(beta0, betaA, delta, A, D, E, I, S)
		didt = dIdt(alpha, gammaI, delta, sigma, E, I)
		dadt = dAdt(gammaA, delta, sigma, A, E)
		drdt = dRdt(gammaA, gammaI, A, I)
		dddt = dDdt(alpha, I)
		
		dsdt, dedt, didt, dadt, drdt, dddt = seiard(z, t, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
		
		d2sdt2 = -beta0 * (S * (didt + betaA * dadt) * (Np - D) + (I + betaA * A) * (dsdt * (Np - D) + S * dddt)) / (Np - D) ** 2
		d2edt2 = -d2sdt2 - delta * dedt 
		d2idt2 = sigma * delta * dedt - gammaI * didt - alpha * didt 
		d2adt2 = (1. - sigma) * delta * dedt - gammaA * dadt 
		d2rdt2 = gammaI * didt + gammaA * dadt 
		d2ddt2 = alpha * didt 
		
		return np.array([d2sdt2, d2edt2, d2idt2, d2adt2, d2rdt2, d2ddt2])
	
	# Jacobian of seiard model...
	def dseiard_(z, t, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np):
		S, E, I, A, R, D = z 
		J = np.zeros((6, 6)).astype(np.float64)
		# S 
		J[0,0] = -beta0 * (I + betaA * A) / (Np - D)
		J[0,2] = -beta0 * S / (Np - D)
		J[0,3] = -beta0 * betaA * A * S / (Np - D)
		J[0,5] = -beta0 * (I + betaA * A) * S / (Np - D) ** 2
		# E 
		J[1,0] = beta0 * (I + betaA * A) / (Np - D)
		J[1,1] = -delta 
		J[1,3] = beta0 * betaA * A * S / (Np - D)
		J[1,5] = beta0 * (I + betaA * A) * S / (Np - D) ** 2
		# I 
		J[2,1] = sigma * delta 
		J[2,2] = -gammaI -alpha 
		# A 
		J[3,1] = (1 - sigma) * delta 
		J[3,3] = gammaA 
		# R 
		J[4,2] = gammaI 
		J[4,3] = gammaA 
		# D 
		J[5,2] = alpha 
		
		return J
	
	# Parameters of the CP problem
	alpha = 1./21.
	beta0 = 2.
	betaA = 0.2
	gammaA = 1./7.
	gammaI = 1./14.
	delta = 1./5.
	sigma = 0.4
	# Numero residenti in Emilia-Romagna al 01-01-2019, fonte ISTAT, portale:
	# http://dati.istat.it/Index.aspx?DataSetCode=DCIS_POPRES1#
	Np = 4459453
	
	# From 01-02-2019 to 15-04-2019
	fname = 'covid19_hospitalized_italy.csv'
	data = pd.read_csv(fname)
	y = data['EmiliaRomagna'].to_numpy().astype(np.float64)
	
	# MEANING:    [S        E   I   A   R   D ]
	z0 = np.array([Np - 1., 1., 0., 0., 0., 0.])
	n = 4
	T = 100 
	h = 2. ** (-n)
	time = np.arange(0., T, step=h)
	
	# Crank-Nicolson 
	begin = process_time()
	# cn = ForwardEuler(seiard, z0, n, T, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
	cn = CrankNicolson(seiard, dseiard, z0, n, T, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
	# cn = odeint(seiard, z0, time, args=(alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np))
	end = process_time()
	print_execution_info(cn, end - begin, implicit = False)
	zcn = cn['solution']
	
	# zcn = odeint(seiard, z0, time, args=(alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np))
	
	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 30))
	fig.suptitle('Crank-Nicolson Approximation of Covid-19 Spread using SEIARD Model', fontsize=30)
	
	titles = ['Susceptible', 'Exposed', 'Symptomatic Infected', 'Asymptomatic Infected', 'Recovered', 'Dead']
	ylabs = [r'$S(t)$', r'$E(t)$', r'$I(t)$', r'$A(t)$', r'$R(t)$', r'$D(t)$']
	
	for i in range(6):
		idx = i//2, i%2 
		ax = axes[idx]
		ax.plot(time, zcn[:,i], label = 'Crank-Nicolson')
		# ax.plot(time[:int(y.shape[0]/h)], zcn[:int(y.shape[0]/h),i], label = 'Crank-Nicolson')
		# if i < 5:
			# ax.plot(np.arange(0, y.shape[0]), y, 'r--o', label = 'Infected people')
		# ax.set_ylim(0., np.max(y) + 10.)
		ax.set_xlabel('time', fontsize=18)
		ax.set_ylabel(ylabs[i], fontsize=18)
		ax.set_title(titles[i], fontsize=18)
	
	plt.savefig('seiard.pdf')
	plt.clf()