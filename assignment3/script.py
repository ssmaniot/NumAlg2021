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
	# print('ForwardEuler(N={}, T={})'.format(N, T))
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
# NO LONGER IN USE
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
def BackwardEuler(f, y0, N, T, tol, *argv):
	# print('BackwardEuler(N={}, T={})'.format(N, T))
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
	
	while i < len(time[:-1]):
		# y[i+1], iter = NewtonRaphson(f_, Df_, y[i])
		y[i+1], iter = VariableSecantScheme(f_, y[i] - 1.e2, y[i] + 1.e2, tol = tol)
		nonlinear_iterations[i] = iter 
		i += 1
	return { 'solution': y, 'time_steps': time.shape[0], 'avg_nonlinear_iter': np.mean(nonlinear_iterations), 'no_nonlinear_convergence': np.sum(nonlinear_iterations == 1000) }

# Crank-Nicolson solver for systems of 1st order ODE
def CrankNicolson(f, y0, N, T, tol, *argv):
	# print('CrankNicolson(N={}, T={})'.format(N, T))
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
	
	while i < len(time[:-1]):
		# y[i+1], iter = NewtonRaphson(f_, Df_, y[i])
		y[i+1], iter = VariableSecantScheme(f_, y[i] - 1.e2, y[i] + 1.e2, tol = tol)
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
	
	print('#########################################################')
	print('###   P A R T   1 :   S P R I N G   E Q U A T I O N   ###')
	print('#########################################################\n\n')
	
	# Function of spring model 
	def spring(z, t, k, m):
		y = z[0]
		v = z[1]
		dydt = v 
		dvdt = -k/m * y 
		return np.array([dydt, dvdt])
	
	# Parameters of the CP problem
	z0 = np.array([0.1, 0.])
	k = 1.
	m = 0.25
	T = 15
	N = 8
	
	# Allocating array for error and elapsed time 
	err = np.ones((N, 4))
	elapsed_time = np.empty((N, 5))
	nonlinear_iter_info = np.empty((N, 5))
	
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle('Comparison Between Numerical and Analytical Solutions', fontsize=30)
	
	for n in range(1, N + 1):
		# print('---')
		h = 2. ** (-n)
		time = np.arange(0., T, step=h)
		elapsed_time[n-1,0] = h
		nonlinear_iter_info[n-1,0] = h 
		
		# Solving using numerical schemes
		
		# Forward Euler 
		begin = process_time()
		fe = ForwardEuler(spring, z0, n, T, k, m)
		end = process_time()
		# print_execution_info(fe, end - begin, implicit = False)
		elapsed_time[n-1,1] = end - begin
		zfe = fe['solution']
		
		# Backward Euler
		begin = process_time()
		be = BackwardEuler(spring, z0, n, T, 1.e-9, k, m)
		end = process_time()
		# print_execution_info(be, end - begin, implicit = True)
		elapsed_time[n-1,2] = end - begin
		nonlinear_iter_info[n-1,1] = be['avg_nonlinear_iter']
		nonlinear_iter_info[n-1,3] = be['no_nonlinear_convergence']
		zbe = be['solution']
		
		# Crank-Nicolson 
		begin = process_time()
		cn = CrankNicolson(spring, z0, n, T, 1.e-9, k, m)
		end = process_time()
		# print_execution_info(cn, end - begin, implicit = True)
		elapsed_time[n-1,3] = end - begin
		nonlinear_iter_info[n-1,2] = cn['avg_nonlinear_iter']
		nonlinear_iter_info[n-1,4] = cn['no_nonlinear_convergence']
		zcn = cn['solution']
		
		# scipy.integrate.odeint()
		# print('scipy.integrate.odeint()')
		begin = process_time()
		zoi = odeint(spring, z0, time, args=(k, m))
		end = process_time()
		elapsed_time[n-1,4] = end - begin
		# print('- Elapsed CPU time:                       {}'.format(end - begin))
		# print('\n')
		
		# Analytical solution
		zan = z0[0] * np.cos(np.sqrt(k / m) * time)
		
		# Error computation
		err[n-1,:] = np.max(np.abs(zfe[1:,0] - zan[1:])), np.max(np.abs(zbe[1:,0] - zan[1:])), np.max(np.abs(zcn[1:,0] - zan[1:])), np.max(np.abs(zoi[1:,0] - zan[1:]))
		
		# Plotting results for N=1 and N=8
		if n == 1 or n == 8:
			idx = 0 if n == 1 else 1
			axes[idx].plot(time, zfe[:,0], '-', label = 'Forward Euler')
			axes[idx].plot(time, zbe[:,0], '--', label = 'Backward Euler')
			axes[idx].plot(time, zcn[:,0], '-', label = 'Crank-Nicolson')
			axes[idx].plot(time, zoi[:,0], '-.', label = 'scipy.integrate.odeint()')
			axes[idx].plot(time, zan, ':', label = 'Analytical Solution')
			axes[idx].set_xlabel(r'time', fontsize=18)
			axes[idx].set_ylabel(r'$y(t)$', fontsize=18)
			axes[idx].set_title('Spring Equation for $N={}$'.format(n), fontsize=18)
			axes[idx].legend()
	
	plt.savefig('numerical_analytical_comp.pdf')
	plt.clf()
	
	models = ['Forward Euler', 'Backward Euler', 'Crank-Nicolson', 'scipy.integrate.odeint()']
	
	# Plotting the convergence of error and error ratio for tol = 1.e-9
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle(r'Convergence of error, tol = $1.$e$-9$', fontsize=30)
	
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
	
	plt.savefig('error1e-9.pdf')
	plt.clf()
	
	# print('---\n\n')
	print(' ' * 13 + 'CPU TIME ELAPSED\n')
	df = pd.DataFrame(data = elapsed_time, columns = ['h'] + models)
	print(df)
	with open('spring_cte.tex', 'w') as f:
		for i in range(len(df)):
			for j in range(len(df.columns)):
				f.write('{:.4e}'.format(df.iat[i,j]))
				if (j < df.shape[1] - 1):
					f.write(' & ')
			f.write(r'\\')
			f.write('\n')
	print('\n\n')
	
	print(' ' * 13 + 'AVG NONLINEAR ITERATIONS' + ' ' * 8 + 'TIME STEPS WITH NO CONVERGENCE\n')
	df = pd.DataFrame(data = nonlinear_iter_info, columns = ['h'] + models[1:3] * 2)
	print(df)
	with open('spring_avg_nl.tex', 'w') as f:
		for i in range(len(df)):
			for j in range(len(df.columns)):
				f.write('{:.4e}'.format(df.iat[i,j]))
				if (j < df.shape[1] - 1):
					f.write(' & ')
			f.write(r'\\')
			f.write('\n')
	print('\n\n')
	
	# Allocating array for error and elapsed time 
	err = np.ones((N, 4))
	
	for n in range(1, N + 1):
		h = 2. ** (-n)
		time = np.arange(0., T, step=h)
		
		# Solving using numerical schemes
		
		# Forward Euler 
		fe = ForwardEuler(spring, z0, n, T, k, m)
		zfe = fe['solution']
		
		# Backward Euler
		be = BackwardEuler(spring, z0, n, T, 1.e-3, k, m)
		zbe = be['solution']
		
		# Crank-Nicolson
		cn = CrankNicolson(spring, z0, n, T, 1.e-3, k, m)
		zcn = cn['solution']
		
		# scipy.integrate.odeint()
		zoi = odeint(spring, z0, time, args=(k, m))
		
		# Analytical solution
		zan = z0[0] * np.cos(np.sqrt(k / m) * time)
		
		# Error computation
		err[n-1,:] = np.max(np.abs(zfe[1:,0] - zan[1:])), np.max(np.abs(zbe[1:,0] - zan[1:])), np.max(np.abs(zcn[1:,0] - zan[1:])), np.max(np.abs(zoi[1:,0] - zan[1:]))
	
	# Plotting the convergence of error and error ratio for tol = 1.e-3
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle(r'Convergence of error, tol = $1.$e$-3$', fontsize=30)
	
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
	
	plt.savefig('error1e-3.pdf')
	plt.clf()
	
	
	"""
	PART 2: EPIDEMIOLOGICAL MODEL (OPTIONAL)
	"""

	print('#########################################################')
	print('###      P A R T   2 :   S E I A R D   M O D E L      ###')
	print('#########################################################\n\n')
	
	# Function of SEIARD model 
	def seiard(z, t, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np):
		S, E, I, A, R, D = z  
		dsdt = -beta0 * (I + betaA * A) * S / (Np - D)
		dedt = -dsdt - delta * E 
		didt = sigma * delta * E - gammaI * I - alpha * I 
		dadt = (1. - sigma) * delta * E - gammaA * A 
		drdt = gammaI * I + gammaA * A
		dddt = alpha * I
		return np.array([dsdt, dedt, didt, dadt, drdt, dddt])
	
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
	N = 4
	T = 100 
	h = 2. ** (-n)
	time = np.arange(0., T, step=h)
	
	# Allocating array for error and elapsed time 
	err = np.empty((N + 1, 3))
	elapsed_time = np.empty((N + 1, 5))
	nonlinear_iter_info = np.empty((N + 1, 5))
	
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle('Comparison Between Numerical and Analytical Solutions', fontsize=30)
	for n in range(0, N + 1):
		# print('---')
		h = 2. ** (-n)
		time = np.arange(0., T, step=h)
		elapsed_time[n,0] = h
		nonlinear_iter_info[n,0] = h
		
		# Solving using numerical schemes
		
		# Forward Euler 
		begin = process_time()
		fe = ForwardEuler(seiard, z0, n, T, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
		end = process_time()
		# print_execution_info(fe, end - begin, implicit = False)
		elapsed_time[n,1] = end - begin
		zfe = fe['solution']
		
		# Backward Euler
		begin = process_time()
		be = BackwardEuler(seiard, z0, n, T, 1.e-9, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
		end = process_time()
		# print_execution_info(be, end - begin, implicit = True)
		elapsed_time[n,2] = end - begin
		nonlinear_iter_info[n,1] = be['avg_nonlinear_iter']
		nonlinear_iter_info[n,3] = be['no_nonlinear_convergence']
		zbe = be['solution']
		
		# Crank-Nicolson 
		begin = process_time()
		cn = CrankNicolson(seiard, z0, n, T, 1.e-9, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
		end = process_time()
		# print_execution_info(cn, end - begin, implicit = True)
		elapsed_time[n,3] = end - begin
		nonlinear_iter_info[n,2] = cn['avg_nonlinear_iter']
		nonlinear_iter_info[n,4] = cn['no_nonlinear_convergence']
		zcn = cn['solution']
		
		# scipy.integrate.odeint()
		# print('scipy.integrate.odeint()')
		begin = process_time()
		zoi = odeint(seiard, z0, time, args=(alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np))
		end = process_time()
		elapsed_time[n,4] = end - begin
		# print('- Elapsed CPU time:                       {}'.format(end - begin))
		# print('\n')
		
		# Error computation
		err[n,:] = np.max(np.abs(zfe[1:] - zoi[1:])), np.max(np.abs(zbe[1:] - zoi[1:])), np.max(np.abs(zcn[1:] - zoi[1:]))
		
		# Plotting results for N=0
		fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 30))
		fig.suptitle('Crank-Nicolson Approximation of Covid-19 Spread using SEIARD Model', fontsize=30)
		titles = ['Susceptible', 'Exposed', 'Symptomatic Infected', 'Asymptomatic Infected', 'Recovered', 'Dead']
		ylabs = [r'$S(t)$', r'$E(t)$', r'$I(t)$', r'$A(t)$', r'$R(t)$', r'$D(t)$']
		for i in range(6):
			idx = i//2, i%2 
			ax = axes[idx]
			ax.plot(time, zfe[:,i], '-', label = 'Forward Euler')
			ax.plot(time, zbe[:,i], '--', label = 'Backward Euler')
			ax.plot(time, zcn[:,i], '-.', label = 'Crank-Nicolson')
			ax.plot(time, zoi[:,i], ':', label = 'scipy.integrate.odeint()')
			ax.set_xlabel('time', fontsize=18)
			ax.set_ylabel(ylabs[i], fontsize=18)
			ax.set_title(titles[i], fontsize=18)
			ax.legend()
	
	plt.savefig('numerical_analytical_comp_seiard.pdf')
	plt.clf()
	
	# print('---\n\n')
	print(' ' * 11 + 'CPU TIME ELAPSED\n')
	df = pd.DataFrame(data = elapsed_time, columns = ['h'] + models)
	print(df)
	with open('seiard_cte.tex', 'w') as f:
		for i in range(len(df)):
			for j in range(len(df.columns)):
				f.write('{:.4e}'.format(df.iat[i,j]))
				if (j < df.shape[1] - 1):
					f.write(' & ')
			f.write(r'\\')
			f.write('\n')
	print('\n\n')
	
	print(' ' * 11 + 'AVG NONLINEAR ITERATIONS' + ' ' * 8 + 'TIME STEPS WITH NO CONVERGENCE\n')
	df = pd.DataFrame(data = nonlinear_iter_info, columns = ['h'] + models[1:3] * 2)
	print(df)
	with open('seiard_avg_nl.tex', 'w') as f:
		for i in range(len(df)):
			for j in range(len(df.columns)):
				f.write('{:.4e}'.format(df.iat[i,j]))
				if (j < df.shape[1] - 1):
					f.write(' & ')
			f.write(r'\\')
			f.write('\n')
	print('\n\n')
	
	# Plotting the convergence of error and error ratio
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle(r'Convergence of error, tol = $1.$e$-9$', fontsize=30)
	
	axes[0].plot(np.arange(0, N + 1), err, ':o')
	axes[0].set_xlabel(r'$N$', fontsize=18)
	axes[0].set_ylabel(r'$e^{(N)}$', fontsize=18)
	axes[0].set_title(r'Absolute Error', fontsize=18)
	axes[0].legend(models)
	
	axes[1].plot(np.arange(1, N + 1), err[1:] / err[:-1], ':o')
	axes[1].set_xlabel(r'$N$', fontsize=18)
	axes[1].set_ylabel(r'$e^{(N)}/e^{(N-1)}$', fontsize=18)
	axes[1].set_title(r'Error Ratio', fontsize=18)
	axes[1].legend(models)
	
	plt.savefig('error1e-9_seiard.pdf')
	plt.clf()
	
	# Number of averted infections 
	idx = int(y.shape[0] / h)
	II = zcn[idx,2] + zcn[idx,3]
	AI = II - y[-1]
	print('Averted infections:', AI)
	
	# Allocating array for error and elapsed time 
	err = np.empty((N + 1, 3))
	
	for n in range(0, N + 1):
		h = 2. ** (-n)
		time = np.arange(0., T, step=h)
		
		# Solving using numerical schemes
		
		# Forward Euler 
		fe = ForwardEuler(seiard, z0, n, T, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
		zfe = fe['solution']
		
		# Backward Euler
		be = BackwardEuler(seiard, z0, n, T, 1.e-3, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
		zbe = be['solution']
		
		# Crank-Nicolson
		cn = CrankNicolson(seiard, z0, n, T, 1.e-3, alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np)
		zcn = cn['solution']
		
		# scipy.integrate.odeint()
		zoi = odeint(seiard, z0, time, args=(alpha, beta0, betaA, gammaA, gammaI, delta, sigma, Np))
		
		# Error computation
		err[n,:] = np.max(np.abs(zfe[1:] - zoi[1:])), np.max(np.abs(zbe[1:] - zoi[1:])), np.max(np.abs(zcn[1:] - zoi[1:]))
	
	# Plotting the convergence of error and error ratio for tol = 1.e-3
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
	fig.suptitle(r'Convergence of error, tol = $1.$e$-3$', fontsize=30)
	
	axes[0].plot(np.arange(0, N + 1), err, ':o')
	axes[0].set_xlabel(r'$N$', fontsize=18)
	axes[0].set_ylabel(r'$e^{(N)}$', fontsize=18)
	axes[0].set_title(r'Absolute Error', fontsize=18)
	axes[0].legend(models)
	
	axes[1].plot(np.arange(1, N + 1), err[1:] / err[:-1], ':o')
	axes[1].set_xlabel(r'$N$', fontsize=18)
	axes[1].set_ylabel(r'$e^{(N)}/e^{(N-1)}$', fontsize=18)
	axes[1].set_title(r'Error Ratio', fontsize=18)
	axes[1].legend(models[:3])
	
	plt.savefig('error1e-3_seiard.pdf')
	plt.clf()