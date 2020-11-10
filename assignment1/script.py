import sympy as sp 
import numpy as np

def err(cur, prev, e):
    """Computes the relative error between two consecutive points cur, prev 
    
    Args:
        cur (float): The current point
        prev (float): The previous point
        e (float): exponent of the denominator 
    
    Returns:
        float: the relative error 
    """
    return np.abs(cur - prev) / (cur ** e)
    
def NewtonRaphson(f, Df, p0, e = 1., max_iter = 1000, tol = 1e-6):
    """Finds the zero of function f using the Newton-Raphson method.
    
    Args:
        f (function): The target function for finding the zero
        Df (function): The first derivative of function f
        p0 (float): The starting point for finding pm 
        e (float): The exponent of the denominator of the error function 
            (default is 1.)
        max_iter (int): The maximum number of iterations allowed before
            early stopping occurs (default is 1000)
        tol (float): The minimum distance allowed between two consecutive
            values of p (default is 1e-6)
    
    Returns:
        dict: A dictionary which contains the following items 
            points (list[double]): the sequence of points computed by 
                the algorithm before reaching convergence/early stopping 
            errors (list[double]): the sequence of error between pair
                of consecutive points 
            iter (int): number of iterations computed by the algorithm
    """
    p = [p0]
    s = []
    
    for k in range(1, max_iter):
        p.append(p[k-1] - f(p[k-1])/Df(p[k-1]))
        s.append(err(p[k], p[k-1], e))
        if s[-1] < tol:
            break
            
    return { 'points': p, 'errors': s, 'iter': k }

def SecantScheme(f, p0, p1, method, e = 1, max_iter = 1000, tol = 1e-6):
    """Finds the zero of function f using secant scheme methods.
    
    Args:
        f (function): The target function for finding the zero
        Df (function): The first derivative of function f
        p0 (float): The first of the initial points
        p1 (float): The second of the initial points 
        method (string): Name of the secant scheme methods; the two 
            supported algorithms are 'fixed' and 'variable'
        e (float): The exponent of the denominator of the error function 
            (default is 1.)
        max_iter (int): The maximum number of iterations allowed before
            early stopping occurs (default is 1000)
        tol (float): The minimum distance allowed between two consecutive
            values of p (default is 1e-6)
    
    Returns:
        dict: A dictionary which contains the following items 
            points (list[double]): the sequence of points computed by 
                the algorithm before reaching convergence/early stopping 
            errors (list[double]): the sequence of error between pair
                of consecutive points 
            iter (int): number of iterations computed by the algorithm
    """
    p = [p0, p1]
    s = []
    
    if method == 'fixed':
        q = (p1 - p0) / (f(p1) - f(p0))
        for k in range(2, max_iter):
            p.append(p[k-1] - q * f(p[k-1]))
            s.append(err(p[k], p[k-1], e))
            if s[-1] < tol:
                break
                
    elif method == 'variable':
        for k in range(2, max_iter):
            q = (p[k-1] - p[k-2]) / (f(p[k-1]) - f(p[k-2]))
            p.append(p[k-1] - q * f(p[k-1]))
            s.append(err(p[k], p[k-1], e))
            if s[-1] < tol:
                break
                
    else:
        raise Exception("option method='{}' not supported".format(method))
        
    return { 'points': p, 'errors': s, 'iter': k }

pm, p0, K, t = sp.symbols(['pm', 'p0', 'K', 't'])
expr = pm / (1 + (pm/p0 - 1) * sp.exp(-K*pm*t))
sf = expr.subs({p0: 100, K: 2 * 1e-6, t: 60}) - 25000
sDf = sp.diff(sf, pm)

f = sp.lambdify(pm, sf, modules=['numpy'])
Df = sp.lambdify(pm, sDf, modules=['numpy'])

res = NewtonRaphson(f, Df, p0 = 30000, e = 1)
res = SecantScheme(f, p0 = 10000, p1 = 50000, method = 'fixed', e = 1.618, max_iter = 9999)
p = res['points']
s = res['errors']
k = res['iter']

print("Last pm:", p[max(len(p) - 5, 0):])
print("Best pm:", p[-1])
print("Last er:", s[max(len(s) - 5, 0):])
print("Number of iterations:", k)

import matplotlib.pyplot as plt
from numpy import linspace 

pT = sp.lambdify(t, expr.subs({p0: 100, pm: p[-1], K: 2 * 1e-6}))
T = linspace(0, 100, 1000)
plt.plot(T, pT(T))
plt.axhline(p[-1], color='r')
#plt.axvline(60, color='g')
plt.show()