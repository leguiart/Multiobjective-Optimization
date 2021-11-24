import numpy as np
import json
import os
import os.path

def save_json(path, analytics):
    with open(path + ".json", 'a') as fp:           
        fp.write('\n')
        json.dump(analytics, fp)

C1 = 10e-4
RHO = .5
ALPHA_INITIAL = 1.
MAX_ITS = 5000

def p(x, grad, w):
    return -grad(x, w)

def grad_sphere(x, w):
    return 2*x

def func_sphere(x, w):
    return np.sum(x**2)

def grad_schaffer(x, w):
    return -np.sin(np.sqrt(np.abs(x)))-(x**2/np.abs(x)*np.cos(np.sqrt(np.abs(x))))/(2*np.sqrt(np.abs(x)))

def func_schaffer(x, w):
    return 418.9829*2 - np.sum(x*np.sin(np.sqrt(np.abs(x))))


"""
DTLZ1
"""
def grad_weighted_dtlz1(x, w):
    gradf1, gradf2, gradf3 = grad_dtlz1(x)
    gradf1 *= w[0]
    gradf2 *= w[1]
    gradf3 *= w[2]
    return gradf1 + gradf2 + gradf3

def grad_dtlz1(x):
    gradf1 = np.append(x[1]*501 + 50*x[1]*((x[2:] - 0.5)**2 - np.cos(20*np.pi*(x[2:] - 0.5))).sum(), 
                        x[0]*501 + 50*x[0]*((x[2:] - 0.5)**2 - np.cos(20*np.pi*(x[2:] - 0.5))).sum())
    gradf1 = np.append(gradf1, 50*x[1]*x[2]*(2*(x[2:] - 0.5) + 20*np.pi*np.sin(x[2:] - 0.5)))
    gradf2 = np.append(0.5 - 0.5 * x[1] + (0.5 - 0.5 * x[1])*g_dtlz1(x), -0.5 * x[0] - 0.5 * x[0] * g_dtlz1(x))
    gradf2 = np.append(gradf2, (0.5 * x[0] - 0.5 * x[0]*x[1])*(100*(2*(x[2:] + 20*np.pi*np.sin(x[2:] - 0.5)))))
    gradf3 = np.append(-.5 - 0.5*(g_dtlz1(x)), 0)
    gradf3 = np.append(gradf3, 0.5*(1 - x[0])*(100*(2*(x[2:] + 20*np.pi*np.sin(x[2:] - 0.5)))))
    return (gradf1, gradf2, gradf3)


def weighted_dtlz1(x, w):
    f1, f2, f3 = func_dtlz1(x)
    f1 *= w[0]
    f2 *= w[1]
    f3 *= w[2]
    return f1 + f2 + f3

def func_dtlz1(x):
    f1 = 0.5 * x[0] * x[1]*(1. + g_dtlz1(x))
    f2 = 0.5 * x[0] * (1 - x[1])*(1. + g_dtlz1(x))
    f3 = 0.5 * (1 - x[0])*(1. + g_dtlz1(x))
    return (f1, f2, f3)

def g_dtlz1(x : np.array):
    return 100.*(10 + ((x[2:] - 0.5)**2 - np.cos(20*np.pi*(x[2:] - 0.5))).sum())

"""
DTLZ2
"""
def grad_weighted_dtlz2(x, w):
    f1, f2, f3 = grad_dtlz2(x)
    f1 *= w[0]
    f2 *= w[1]
    f3 *= w[2]
    return f1 + f2 + f3

def grad_dtlz2(x):
    dg = 2*(x[2:] - 0.5)
    gradf1 = np.append(-np.pi/2*np.sin(np.pi/2*x[0])*np.cos(np.pi/2*x[1])*(1 * g_dtlz2(x)), -np.pi/2*np.sin(np.pi/2*x[1])*np.cos(np.pi/2*x[0])*(1 + g_dtlz2(x)))
    gradf1 = np.append(gradf1, np.cos(np.pi/2*x[0])*np.cos(np.pi/2*x[1])*dg)
    gradf2 = np.append(-np.pi/2*np.sin(np.pi/2*x[0])*np.sin(np.pi/2*x[1])*(1 + g_dtlz2(x)), np.pi/2*np.cos(np.pi/2*x[0])*np.cos(np.pi/2*x[1])*(1 + g_dtlz2(x)))
    gradf2 = np.append(gradf2, np.cos(np.pi*x[0])*np.sin(np.pi*x[1])*dg)
    gradf3 = np.append(np.pi/2*np.cos(np.pi*x[0])*(1 + g_dtlz2(x)), 0)
    gradf3 = np.append(gradf3, np.sin(np.pi/2*x[0])*dg)
    return (gradf1, gradf2, gradf3)

def weighted_dtlz2(x, w):
    f1, f2, f3 = func_dtlz2(x)
    f1 *= w[0]
    f2 *= w[1]
    f3 *= w[2]
    return f1 + f2 + f3

def func_dtlz2(x):
    f1 = np.cos(0.5*np.pi*x[0])*np.cos(0.5*np.pi*x[1])*(1 + g_dtlz2(x))
    f2 = np.cos(0.5*np.pi*x[0])*np.sin(0.5*np.pi*x[1])*(1 + g_dtlz2(x))
    f3 = np.sin(0.5*np.pi*x[0])*(1 + g_dtlz2(x))
    return (f1, f2, f3)

def g_dtlz2(x):
    return ((x[2:] - 0.5)**2 ).sum()

def backtracking(x, f, p, grad, alpha_initial, w):
    alpha = alpha_initial
    # count  = 0
    while not (f(x + alpha * p(x, grad, w), w) <= f(x, w) + C1 * alpha * grad(x, w).T @ p(x, grad, w)):
        # count += 1
        alpha *= RHO
    return alpha

def gradient_descent(f, p, grad, x_0, alpha_initial, max_its, w):
    x_k = x_0
    its = 0
    alpha_k = alpha_initial
    epsilon = 0.75
    while its < max_its:
        alpha_k = backtracking(x_k, f, p, grad, alpha_k, w)
        g = alpha_k*p(x_k, grad, w)
        x_k += g

        if (x_k < 0.).any() or (x_k > 1.).any():
            minus = np.where(x_k < 0)[0]
            plus = np.where(x_k > 1.)[0]
            for i in minus:
                x_k[i] -= g[i]
            for i in plus:
                x_k[i] -= g[i]
        its += 1

    return x_k

def stop(its, x_k, f, max_its, tol):
    return its < max_its and f(x_k) > tol

weights = [[0.1666, 0.6666, 0.1666],
           [0.8,0.1,0.1],  
           [0.1,0.8,.1], 
           [0.1, 0.1, 0.8],
           [0.2, 0., 0.8], 
           [0.2, 0.8, 0.],
           [0.8, 0.2, 0.0],
           [0.333, 0.333, 0.333], 
           [0.25, 0.25, 0.5], 
           [0.25, 0.5, 0.25], 
           [0.5, 0.25, 0.25],
           [0.5,0.5,0.], 
           [0,0.5,0.5], 
           [0.5,0,0.5], 
           [0.2,0.2,0.6], 
           [0.2, 0.6, 0.2], 
           [0.6, 0.2, 0.2], 
           [0.4, 0.3, 0.3], 
           [0.3, 0.4, 0.3],           
           [0.3, 0.3, 0.4]]
analytics = {"DTLZ1-WS" : {"evaluations" : [], "phenotypes":[]}, "DTLZ2-WS" : {"evaluations" : [], "phenotypes":[]} }
dtlz1_solutions = [[]]
dtlz1_variables = [[]]
dtlz2_solutions = [[]]
dtlz2_variables = [[]]

print("Starting DLZ1-WS Optimization")
for j, weight in enumerate(weights):
    print(f"DLZ1-WS Optimization with weights: {weight}")
    for i in range(100):
        x = gradient_descent(weighted_dtlz1, p, grad_weighted_dtlz1, np.random.uniform(0., 1., size = 12), 1., 1000, weight)
        dtlz1_solutions[j] += [list(func_dtlz1(x))]
        dtlz1_variables[j] += [list(x)]
    dtlz1_solutions += [[]]
    dtlz1_variables += [[]]

analytics["DTLZ1-WS"]["evaluations"] = dtlz1_solutions
analytics["DTLZ1-WS"]["phenotypes"] = dtlz1_variables

print("Starting DLZ2-WS Optimization")
for j, weight in enumerate(weights):
    print(f"DLZ2-WS Optimization with weights: {weight}")
    for i in range(100):
        x = gradient_descent(weighted_dtlz2, p, grad_weighted_dtlz2, np.random.uniform(0., 1., size = 12), 1., 1000, weight)
        dtlz2_solutions[j] += [list(func_dtlz2(x))]
        dtlz2_variables[j] += [list(x)]
    dtlz2_solutions += [[]]
    dtlz2_variables += [[]]

analytics["DTLZ2-WS"]["evaluations"] = dtlz2_solutions
analytics["DTLZ2-WS"]["phenotypes"] = dtlz2_variables

save_json('ws-analytics', analytics)