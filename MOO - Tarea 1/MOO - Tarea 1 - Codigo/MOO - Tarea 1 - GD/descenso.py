import numpy as np
import json
import os
import os.path

C1 = 10e-4
RHO = .5
ALPHA_INITIAL = 1.
MAX_ITS = 5000

def p(x, grad):
    return -grad(x)

def grad_sphere(x):
    return 2*x

def func_sphere(x):
    return np.sum(x**2)

def grad_schaffer(x):
    return -np.sin(np.sqrt(np.abs(x)))-(x**2/np.abs(x)*np.cos(np.sqrt(np.abs(x))))/(2*np.sqrt(np.abs(x)))

def func_schaffer(x):
    return 418.9829*2 - np.sum(x*np.sin(np.sqrt(np.abs(x))))

def backtracking(x, f, p, grad, alpha_initial):
    alpha = alpha_initial
    while not (f(x + alpha * p(x, grad)) <= f(x) + C1 * alpha * grad(x).T @ p(x, grad)):
        alpha *= RHO
    return alpha

def gradient_descent(f, p, grad, x_0, alpha_initial, max_its, tol, analytics):
    x_k = x_0
    its = 0
    alpha_k = alpha_initial
    while its < max_its and f(x_k) > tol:
        analytics["x_s"]+=[list(x_k)]
        analytics["evaluations"]+=[f(x_k)]
        analytics["gradients"]+=[list(grad(x_k))]
        analytics["alphas"]+=[alpha_k]
        alpha_k = backtracking(x_k, f, p, grad, alpha_k)
        x_k += alpha_k*p(x_k, grad)
        its += 1

    return x_k

def stop(its, x_k, f, max_its, tol):
    return its < max_its and f(x_k) > tol
    
sphere_analytics = {"x_s":[], "evaluations":[], "gradients":[], "alphas":[]}
print(gradient_descent(func_sphere, p, grad_sphere, np.random.uniform(-5., 5., size = 2), 3., 5000, 0.0001, sphere_analytics))
schaffer_analytics = {"x_s":[], "evaluations":[], "gradients":[], "alphas":[]}
print(gradient_descent(func_schaffer, p, grad_schaffer, np.random.uniform(-500., 500., size = 2), .3, 5000, 0.001, schaffer_analytics))

if os.path.isfile('sphere_analytics.json'):
    with open('sphere_analytics.json', 'a') as fp:           
        fp.write('\n')
        json.dump(sphere_analytics, fp)
else:
    with open('sphere_analytics.json', 'w') as fp: 
        json.dump(sphere_analytics, fp)

if os.path.isfile('schaffer_analytics.json'):
    with open('schaffer_analytics.json', 'a') as fp:           
        fp.write('\n')
        json.dump(schaffer_analytics, fp)
else:
    with open('schaffer_analytics.json', 'w') as fp: 
        json.dump(schaffer_analytics, fp)