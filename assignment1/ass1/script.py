import sympy as sp 
import numpy as np

def err(cur, prev):
    """Computes the relative error between two consecutive points cur, prev 
    
    Args:
        cur (float): The current point
        prev (float): The previous point
        e (float): exponent of the denominator 
    
    Returns:
        float: the relative error 
    """
    return np.abs((cur - prev) / cur)
    
def NewtonRaphson(f, Df, p0, max_iter = 1000, tol = 1e-6):
    """Finds the zero of function f using the Newton-Raphson method.
    
    Args:
        f (function): The target function for finding the zero
        Df (function): The first derivative of function f
        p0 (float): The starting point for finding pm 
        max_iter (int): The maximum number of iterations allowed before
            early stopping occurs (default is 1000)
        tol (float): The minimum distance allowed between two consecutive
            values of p (default is 1e-6)
    
    Returns:
        dict: A dictionary which contains the following items 
            points (ndarray[double]): the sequence of points computed by 
                the algorithm before reaching convergence/early stopping 
            errors (ndarray[double]): the sequence of error between pair
                of consecutive points 
            iter (int): number of iterations computed by the algorithm
			stops (ndarray[int]): iterations at which the algorithm stops 
				at tolerance 1e-2, 1e-4 and 1e-6. It holds 3 elements.
    """
    p = [p0]
    s = []
    stops = []
    tols = [1.e-2, 1.e-4, 1.e-6]
    tol = 0
	
    for k in range(1, max_iter):
        p.append(p[k-1] - f(p[k-1])/Df(p[k-1]))
        s.append(err(p[k], p[k-1]))
        while s[-1] < tols[tol] and tol < 2:
            stops.append(tol)
            tol += 1
        if s[-1] < tols[-1]:
            break
            
    return { 'points': np.array(p), 'errors': np.array(s), 'iter': k, 'stops': np.array(stops) }

def SecantScheme(f, p0, p1, method, max_iter = 1000, tol = 1e-6):
    """Finds the zero of function f using secant scheme methods.
    
    Args:
        f (function): The target function for finding the zero
        Df (function): The first derivative of function f
        p0 (float): The first of the initial points
        p1 (float): The second of the initial points 
        method (string): Name of the secant scheme methods; the two 
            supported algorithms are 'fixed' and 'variable'
        max_iter (int): The maximum number of iterations allowed before
            early stopping occurs (default is 1000)
        tol (float): The minimum distance allowed between two consecutive
            values of p (default is 1e-6)
    
    Returns:
        dict: A dictionary which contains the following items 
            points (ndarray[double]): the sequence of points computed by 
                the algorithm before reaching convergence/early stopping 
            errors (ndarray[double]): the sequence of error between pairs
                of consecutive points 
            iter (int): number of iterations computed by the algorithm
			stops (ndarray[int]): iterations at which the algorithm stops 
				at tolerance 1e-2, 1e-4 and 1e-6. It holds 3 elements.
    """
    p = [p0, p1]
    s = []
    stops = []
    tols = [1.e-2, 1.e-4, 1.e-6]
    tol = 0
    
    if method == 'fixed':
        q = (p1 - p0) / (f(p1) - f(p0))
        for k in range(1, max_iter):
            p.append(p[k] - q * f(p[k]))
            s.append(err(p[k+1], p[k]))
            while s[-1] < tols[tol] and tol < 2:
                stops.append(tol)
                tol += 1
            if s[-1] < tols[-1]:
                break
                
    elif method == 'variable':
        for k in range(1, max_iter):
            q = (p[k] - p[k-1]) / (f(p[k]) - f(p[k-1]))
            p.append(p[k] - q * f(p[k]))
            s.append(err(p[k+1], p[k]))
            while s[-1] < tols[tol] and tol < 2:
                stops.append(tol)
                tol += 1
            if s[-1] < tols[-1]:
                break
                
    else:
        raise Exception("option method='{}' not supported".format(method))
        
    return { 'points': np.array(p), 'errors': np.array(s), 'iter': k, 'stops': np.array(stops) }

# Use symbolic computation to compute the first derivative of f(t)
pm, p0, K, t = sp.symbols(['pm', 'p0', 'K', 't'])
expr = pm / (1 + (pm/p0 - 1) * sp.exp(-K*pm*t))
sf = expr.subs({p0: 100, K: 2 * 1e-6, t: 60}) - 25000
sDf = sp.diff(sf, pm)

# Convert the symbolic representation to a numpy function 
f = sp.lambdify(pm, sf, modules=['numpy'])
Df = sp.lambdify(pm, sDf, modules=['numpy'])

nr = NewtonRaphson(f, Df, p0 = 30000)
fss = SecantScheme(f, p0 = 30000, p1 = 70000, method = 'fixed')
vss = SecantScheme(f, p0 = 30000, p1 = 70000, method = 'variable')
print("# of iterations for each algorithm")
print(" - Newton-Raphson: {}".format(nr['iter']))
print("   stops: {}".format(nr['stops']))
print(" - Fixed Secant Scheme: {}".format(fss['iter']))
print("   stops: {}".format(fss['stops']))
print(" - Variable Secant Scheme: {}".format(vss['iter']))
print("   stops: {}".format(fss['stops']))

import matplotlib.pyplot as plt

rel_err_nr  = [None] * 3
rel_err_fss = [None] * 3
rel_err_vss = [None] * 3

for i, e in enumerate([1., 1.618, 2.]):
    plt.title("$s_k/s_k^p$ for $p = {}$".format(e))
    plt.xlabel('Iteration')
    plt.ylabel(r'$s_k/s_{k-1}^{' + '{}'.format(e) + '}$')
    rel_err_nr[i] = np.array([nr['errors'][k+1]/nr['errors'][k]**e for k in range(nr['iter'] - 1)])
    rel_err_fss[i] = np.array([fss['errors'][k+1]/fss['errors'][k]**e for k in range(fss['iter'] - 1)])
    rel_err_vss[i] = np.array([vss['errors'][k+1]/vss['errors'][k]**e for k in range(vss['iter'] - 1)])
    nr_line, = plt.plot(np.arange(1, rel_err_nr[i].shape[0] + 1), rel_err_nr[i], 'o', markerfacecolor='none', ls='--', lw=1)
    fss_line, = plt.plot(np.arange(1, rel_err_fss[i].shape[0] + 1), rel_err_fss[i], 'x', markerfacecolor='none', ls='--', lw=1)
    vss_line, = plt.plot(np.arange(1, rel_err_vss[i].shape[0] + 1), rel_err_vss[i], '*', markerfacecolor='none', ls='--', lw=1)
    legend = plt.legend(handles = [nr_line, fss_line, vss_line], 
                        labels = ['Newton-Raphson', 'Fixed Secant Scheme', 'Variable Secant Scheme'], 
                        loc = 'upper right' if e == 1. else 'upper left')
    plt.xlim([1, 6])
    plt.ylim([-1, 30])
    plt.savefig("compare_model_{}.pdf".format(e))
    plt.clf()

def plot_error_comparison(plt, data, algorithm, mec, loc):
    plt.title("{} $s_k/s_k^p$ comparison".format(algorithm))
    plt.xlabel('Iteration')
    plt.ylabel(r'$s_k/s_{k-1}^p$')
    lines = list()
    markers = ['+', '^', 'p']
    for i, e in enumerate([1., 1.618, 2.]):
        rel_err = np.array([data['errors'][k+1]/data['errors'][k]**e for k in range(data['iter'] - 1)])
        line, = plt.plot(np.arange(1, rel_err.shape[0] + 1), rel_err, markers[i], mec=mec, mfc='none', ls=':', color='k', lw=1)
        lines.append(line)
    s1 = plt.axvline(data['stops'][0], ls=':', lw=1, c='y')
    lines.append(s1)
    s2 = plt.axvline(data['stops'][1], ls=':', lw=1, c='g')
    lines.append(s2)
    legend = plt.legend(handles = lines, labels = ['d = 1', 'd = 1.618', 'd = 2', r'stop $\tau = 1e-2$', r'stop $\tau = 1e-4$'], loc = 'upper right')

plot_error_comparison(plt, nr, "Newton-Raphson", 'r', 'upper right')
plt.savefig("nr.pdf")
plt.clf()
plot_error_comparison(plt, fss, "Fixed Secant Scheme", 'm', 'upper left')
plt.savefig("fss.pdf")
plt.clf()
plot_error_comparison(plt, vss, "Variable Secant Scheme", 'b', 'upper left')
plt.savefig("vss.pdf")
plt.clf()

# Generate the table

def aux(f, model, err, i):
	if i == 0:
		f.write(" & {:.4e} & -- & -- & --".format(model['points'][i]))
		return 
	if i >= model['iter']:
		f.write(" & -- & -- & -- & --")
		return 
	f.write(" & {:.3e}".format(model['points'][i]))
	for m in range(3):
		#print(err[m].shape, i, i-1)
		f.write(" & {:.4e}".format(err[m][i-1]))

k = max([nr['iter'], fss['iter'], vss['iter']])
with open("table.tex", 'w') as f:
	for i in range(k):
		f.write("{} ".format(i))
		aux(f, nr, rel_err_nr, i)
		aux(f, fss, rel_err_fss, i)
		aux(f, vss, rel_err_vss, i)
		f.write(r"\\" + '\n')
		f.write(r"\hline" + '\n')

"""
p = nr['points']

import matplotlib.pyplot as plt
from numpy import linspace 

pT = sp.lambdify(t, expr.subs({p0: 100, pm: p[-1], K: 2 * 1e-6}))
T = linspace(0, 100, 1000)
plt.plot(T, pT(T))
plt.axhline(p[-1], color='r')
plt.show()
"""