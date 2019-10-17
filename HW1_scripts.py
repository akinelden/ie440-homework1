# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:46:53 2019

@author: akin
"""
import numpy as np
from sympy import Symbol, cos, sin, lambdify
import matplotlib.pyplot as plt
import pandas as pd

#%%

x = Symbol('x')
function = x**3*cos(x)**2*sin(x)+3*x**2*cos(x)-5*x
first_deriv = function.diff(x)
second_deriv = first_deriv.diff(x)

f = lambdify(x, function, 'numpy')
df = lambdify(x, first_deriv, 'numpy')
ddf = lambdify(x, second_deriv, 'numpy')

#%%
def plotGraphWithLines(x_pts=[], colors=[], labels=[]):
    t1 = np.arange(-3, 9.05, 0.05)
    plt.figure()
    plt.plot(t1, f(t1), 'b-', label='f(x)')
    for i in range(len(x_pts)):
        plt.axvline(x_pts[i], color=colors[i], label=labels[i])
    plt.legend()
    #plt.savefig("graph.png")
    
plotGraphWithLines()
#%%
def c_rate(x, degree=1):
    res = [None]
    for i in range(1,(len(x) -1)):
        res.append(np.abs(x[i+1] - x[i])/np.abs(x[i] - x[i-1])**degree)
    return res
#%%
def BisectionMethod(a=-3,b=9,epsilon=0.001) :
    iteration=0
    res = []
    while (b - a) >= epsilon:
        x_1 = (a + b) / 2
        fx_1 = f(x_1)
        res.append([iteration, a, b, x_1, fx_1])
        if f(x_1 + epsilon) <= fx_1:
            a = x_1
        else:
            b = x_1
        iteration+=1
    x_star = (a+b)/2
    fx_star = f(x_star)
    res.append([iteration, a, b, x_star, fx_star])
    result_table = pd.DataFrame(res, columns=['iteration' ,'a', 'b', 'x', 'f(x)'])
    result_table['c_rate'] = pd.Series(c_rate(result_table.x))
    return x_star, fx_star, result_table

#%%
def GoldenSection(a,b,epsilon):
    golden_ratio = (1+np.sqrt(5))/2
    gama = 1/golden_ratio
    iteration = 0
    x_1 = b - gama*(b-a)
    x_2 = a + gama*(b-a)
    fx_1 = f(x_1)
    fx_2 = f(x_2)
    res = [[iteration, a, b, x_1, x_2, fx_1, fx_2]]
    while (b-a) >= epsilon:
        iteration+=1
        if(fx_1 >= fx_2):
            a = x_1
            x_1 = x_2
            x_2 = a + gama*(b-a)
            fx_1 = fx_2
            fx_2 = f(x_2)
        else:
            b = x_2
            x_2 = x_1
            x_1 = b - gama*(b-a)
            fx_2 = fx_1
            fx_1 = f(x_1)
        res.append([iteration, a, b, x_1, x_2, fx_1, fx_2])
    result_table = pd.DataFrame(res, columns = ['iteration', 'a', 'b', 'x', 'y', 'f(x)', 'f(y)'])
    result_table['c_rate'] = pd.Series(c_rate((result_table.a + result_table.b)/2))
    x_star = (a+b)/2
    fx_star = f(x_star)
    return x_star, fx_star, result_table

#%%
def NewtonsMethod(x_0, epsilon):
    iteration = 0
    dfx0 = df(x_0)
    ddfx0 = ddf(x_0)
    res = [[iteration, x_0, f(x_0), dfx0, ddfx0]]
    x_1 = x_0-dfx0/ddfx0
    while abs(x_0-x_1)>=epsilon:
        iteration +=1
        x_0 = x_1
        dfx0 = df(x_0)
        ddfx0 = ddf(x_0)
        x_1 = x_0-dfx0/ddfx0
        res.append([iteration, x_0, f(x_0), dfx0, ddfx0])
    res.append([iteration+1, x_1, f(x_1), df(x_1), ddf(x_1)])
    result_table = pd.DataFrame(res, columns = ['iteration', 'x', 'f(x)', "f'(x)", "f''(x)"])
    result_table['c_rate'] = pd.Series(c_rate(result_table.x, 2))
    x_star = x_1
    fx_star = f(x_1)
    return x_star, fx_star, result_table
    
#%%
def NewtonsMethod2(x_0, epsilon):
    iteration = 0
    res = []
    while True:
        dfx0 = df(x_0)
        ddfx0 = ddf(x_0)
        x_1 = x_0-dfx0/ddfx0
        res.append([iteration, x_0, f(x_0), dfx0, ddfx0])
        iteration +=1
        if abs(x_0-x_1)<epsilon:
            break
        else: 
            x_0 = x_1
    res.append([iteration, x_1, f(x_1), df(x_1), ddf(x_1)])
    result_table = pd.DataFrame(res, columns = ['iteration', 'x', 'f(x)', "f'(x)", "f''(x)"])
    result_table['c_rate'] = pd.Series(c_rate(result_table.x, 2))
    x_star = x_1
    fx_star = f(x_1)
    return x_star, fx_star, result_table
#%%
def SecantMethod(x_0, x_1, epsilon):
    iteration = 0
    res = [[iteration, x_0, f(x_0), df(x_0)]]
    iteration += 1
    while True:
        dfx0 = df(x_0)
        dfx1 = df(x_1)
        x_next = x_1 - dfx1 / (dfx1 - dfx0) * (x_1 - x_0)
        res.append([iteration, x_1, f(x_1), dfx1])
        iteration +=1
        if abs(x_next-x_1)<epsilon:
            break
        x_0 = x_1
        x_1 = x_next
    res.append([iteration, x_next, f(x_next), df(x_next)])
    result_table = pd.DataFrame(res, columns = ['iteration', 'x', 'f(x)', "f'(x)"])
    result_table['c_rate'] = pd.Series(c_rate(result_table.x, 1.618))
    x_star = x_next
    fx_star = f(x_next)
    return x_star, fx_star, result_table
#%%

x_star, fx_star, res = BisectionMethod(-3,9,0.001)
plotGraphWithLines([x_star,-3,9],['r','g','g'],['x*','a','b'])
print(res.to_latex(index=False,float_format='%.4f'))

x_star, fx_star, res = GoldenSection(-3,9,0.001)
x_star, fx_star, res = NewtonsMethod(4,0.001)

x_star, fx_star, res = NewtonsMethod(3,0.001)
x_star, fx_star, res2 = NewtonsMethod2(3,0.001)

x_star, fx_star, res = SecantMethod(3,3.1,0.001)

