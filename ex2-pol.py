import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + (np.sin(9*x))**2

n=8
xi = [i/n for i in range(n+1)]
yi = [f(xi[i]) for i in range(n+1)]

#tabela das diferenças divididas:
T = [yi]
for j in range(n+1):
    T += [[(T[j][i+1]-T[j][i])/(xi[i+1+j]-xi[i]) for i in range(n-j)]]

#polinómio interpolador:
def p(x):
    return T[0][0] + T[1][0]*(x-xi[0]) + T[2][0]*(x-xi[0])*(x-xi[1]) + T[3][0]*(x-xi[0])*(x-xi[1])*(x-xi[2]) + T[4][0]*(x-xi[0])*(x-xi[1])*(x-xi[2])*(x-xi[3]) + T[5][0]*(x-xi[0])*(x-xi[1])*(x-xi[2])*(x-xi[3])*(x-xi[4]) + T[6][0]*(x-xi[0])*(x-xi[1])*(x-xi[2])*(x-xi[3])*(x-xi[4])*(x-xi[5]) + T[7][0]*(x-xi[0])*(x-xi[1])*(x-xi[2])*(x-xi[3])*(x-xi[4])*(x-xi[5])*(x-xi[6]) + T[8][0]*(x-xi[0])*(x-xi[1])*(x-xi[2])*(x-xi[3])*(x-xi[4])*(x-xi[5])*(x-xi[6])*(x-xi[7])

#gráficos:
x = np.arange  ( start = 0
                , stop = 1.001
                , step = 0.001
                )

y = p(x)
ff = f(x)
plt.plot(x,ff, label='f(x)')
plt.plot(x, y, 'r-', label='p(x)')

pontos = np.array([ [xi[i],yi[i]] for i in range(n+1) ])
xx, yy = pontos.T
plt.scatter(xx,yy, color='k')

plt.axis([0, 1, -1.6, 1.8])
#plt.axvline(0, color='k')
#plt.axhline(0, color='k')
plt.grid()
plt.legend()
plt.show()
plt.savefig('ex2-pol.pdf', bbox_inches='tight')