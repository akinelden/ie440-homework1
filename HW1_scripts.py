# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:46:53 2019
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
def plotGraphWithLines(x_pts=[], colors=[], labels=[], name="graph"):
    t1 = np.arange(-3, 9.05, 0.05)
    plt.figure()
    plt.plot(t1, f(t1), 'b-', label='f(x)')
    for i in range(len(x_pts)):
        plt.axvline(x_pts[i], color=colors[i], label=labels[i])
    plt.legend()
    plt.savefig("{0}.png".format(name))
    
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
    result_table["log_c_rate"] = -np.log(result_table.c_rate)
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
    result_table["log_c_rate"] = -np.log(result_table.c_rate)
    x_star = (a+b)/2
    fx_star = f(x_star)
    return x_star, fx_star, result_table

#%%
def NewtonsMethod(a, b, x_0, epsilon):
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
        if x_1<a or x_1>b:
            print("Error: The Newton\'s method is not able to find any local minimum in the given range")
            break
    res.append([iteration, x_1, f(x_1), df(x_1), ddf(x_1)])
    result_table = pd.DataFrame(res, columns = ['iteration', 'x', 'f(x)', "f'(x)", "f''(x)"])
    result_table['c_rate'] = pd.Series(c_rate(result_table.x, 2))
    x_star = x_1
    fx_star = f(x_1)
    return x_star, fx_star, result_table
#%%
def SecantMethod(x_0, x_1, epsilon, a=-3, b=9):
    iteration = 0
    res = [[iteration, x_0, f(x_0), df(x_0)]]
    iteration += 1
    dfx0 = df(x_0)
    while True:
        dfx1 = df(x_1)
        x_next = x_1 - dfx1 / (dfx1 - dfx0) * (x_1 - x_0)
        res.append([iteration, x_1, f(x_1), dfx1])
        iteration +=1
        if abs(x_next-x_1)<epsilon:
            break
        x_0 = x_1
        dfx0 = dfx1
        x_1 = x_next
        if x_next<a or x_next>b:
            print("Error: The Secant method is not able to find any local minimum in the given range")
            break
    res.append([iteration, x_next, f(x_next), df(x_next)])
    result_table = pd.DataFrame(res, columns = ['iteration', 'x', 'f(x)', "f'(x)"])
    result_table['c_rate'] = pd.Series(c_rate(result_table.x, 1.618))
    x_star = x_next
    fx_star = f(x_next)
    return x_star, fx_star, result_table
#%%
'''
Bisection Model-1
'''
a=-3
b=9
x_star, fx_star, res = BisectionMethod(a,b,0.001)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,4)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))
#%%
'''
Bisection Model-2
'''
a=-2
b=8
x_star, fx_star, res = BisectionMethod(a,b,0.001)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,4)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))
#%%
'''
Bisection Model-3
'''
a=2
b=9
x_star, fx_star, res = BisectionMethod(a,b,0.001)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,4)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))
#%%
'''
Bisection Model-4
'''
a=2.1
b=9
x_star, fx_star, res = BisectionMethod(a,b,0.001)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,4)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))

#%%
'''
Golden Section Model -2 
'''
a=3.5
b=9
epsilon=0.001
x_star, fx_star, res = GoldenSection(a,b,epsilon)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,4)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))
round(x_star,4),round(fx_star,4)
#%%
'''
Golden Section Model - 3

'''
a=-3
b=9
epsilon=0.669
x_star, fx_star, res = GoldenSection(a,b,epsilon)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,4)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))
round(x_star,4),round(fx_star,4)
#%%
'''
Golden Section Model 1
'''
a=-3
b=9
epsilon=0.001
x_star, fx_star, res = GoldenSection(a,b,epsilon)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,3)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))
round(x_star,4),round(fx_star,4)
#%%
'''
Golden Section Model -4
'''
a=3.80
b=8
epsilon=0.001
x_star, fx_star, res = GoldenSection(a,b,epsilon)
lines = [x_star,a,b]
colors = ['r','g','g']
labels = ['x*','a,b','']
for i in range(min(len(res.x)-1,2)):
    lines.append(res.x[i])
    colors.append('y')
    labels.append('')
labels[-1] = 'intermediary steps'
plotGraphWithLines(lines,colors,labels)
#print(res.to_latex(index=False,float_format='%.4f'))
round(x_star,4),round(fx_star,4)
#%%
'''
Secant Method
'''
secant_par = [[6 ,7, 0.001], [1, 2, 0.005], [3, 4, 0.0001]]
a= -3
b= 9
for par in secant_par:
    x_star, fx_star, res = SecantMethod(par[0],par[1],par[2])
    lines = [x_star,a,b]
    colors = ['r','g','g']
    labels = ['x*','a,b','']
    for i in range(min(len(res.x)-1,2)):
        lines.append(res.x[i])
        colors.append('y')
        labels.append('')
    labels[-1] = 'intermediary steps'
    plotGraphWithLines(lines,colors,labels, 'secant' + str(par[0]))
    print(res.to_latex(index=False,float_format='%.4f'))
    print(round(x_star,4),round(fx_star,4))
#%%
'''
Newton's Method
'''
newton_par = [[3.5, 0.001], [6.6, 0.0001], [7.5, 0.001], [7.6, 0.005]]
a= -3
b= 9
for par in newton_par:
    x_star, fx_star, res = NewtonsMethod(a, b, par[0], par[1])
    lines = [x_star,par[0],a,b]
    colors = ['r','y','g','g']
    labels = ['x*','x0','a,b','']
    plotGraphWithLines(lines,colors,labels, 'newton' + str(par[0]))
    #print(res.to_latex(index=False,float_format='%.4f'))
    print(round(x_star,4),round(fx_star,4),round(ddf(x_star),4))

