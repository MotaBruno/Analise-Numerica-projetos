import numpy as np
import matplotlib.pyplot as plt

n=5
xi=[0,1,2,2.5,3,4]
yi=[1.4,0.6,1.0,0.6,0.6,1.0]

#colunas da tabela de diferenças divididas:
dif1 = [(yi[i+1]-yi[i])/(xi[i+1]-xi[i]) for i in range(n)]
dif2 = [(dif1[i+1]-dif1[i])/(xi[i+2]-xi[i]) for i in range(n-1)]
dif3 = [(dif2[i+1]-dif2[i])/(xi[i+3]-xi[i]) for i in range(n-2)]
dif4 = [(dif3[i+1]-dif3[i])/(xi[i+4]-xi[i]) for i in range(n-3)]
dif5 = [(dif4[i+1]-dif4[i])/(xi[i+5]-xi[i]) for i in range(n-4)]

#polinómio interpolador:
def p(x):
    return yi[0] + dif1[0]*(x-xi[0]) + dif2[0]*(x-xi[0])*(x-xi[1]) + dif3[0]*(x-xi[0])*(x-xi[1])*(x-xi[2]) + dif4[0]*(x-xi[0])*(x-xi[1])*(x-xi[2])*(x-xi[3]) + dif5[0]*(x-xi[0])*(x-xi[1])*(x-xi[2])*(x-xi[3])*(x-xi[4])

#sistema dos Ms para o spline:
A = np.array([[1/6,2/3,1/6,0,0,0], [0,1/6,1/2,1/12,0,0], [0,0,1/12,1/3,1/12,0],[0,0,0,1/12,1/2,1/6],[1,0,0,0,0,0],[0,0,0,0,0,1]])
B = np.array([1.2,-1.2,0.8,0.4,0,0])
M = np.linalg.solve(A,B)

#função spline
def s(x):
    if 0<=x<1:
        return 1/6*M[1]*x**3 + 1.4*(1-x) + (0.6 - M[1]/6)*x

    if x<2:
        return 1/6*M[1]*(2-x)**3 + 1/6*M[2]*(x-1)**3 + (0.6-M[1]/6)*(2-x) + (1.0 - M[2]/6)*(x-1)

    if x<2.5:
        return 1/3*M[2]*(2.5-x)**3 + 1/3*M[3]*(x-2)**3 + (1.0-M[2]/24)*(2.5-x)*2 + (0.6 - M[3]/24)*(x-2)*2

    if x<3:
        return 1/3*M[3]*(3-x)**3 + 1/3*M[4]*(x-2.5)**3 + (0.6-M[3]/24)*(3-x)*2 + (0.6 - M[4]/24)*(x-2.5)*2

    if x<4:
        return 1/6*M[4]*(4-x)**3 + (0.6-M[4]/6)*(4-x) + x-3

S = np.vectorize(s)

#gráficos:
x = np.arange  ( start = 0
                , stop = 4.01
                , step = 0.01
                )

y = p(x)
y1 = S(x)
plt.plot(x, y, label='p(x)')
plt.plot(x, y1, 'r-', label='s(x)')

pontos = np.array([ [xi[i],yi[i]] for i in range(n+1) ])
xx, yy = pontos.T
plt.scatter(xx,yy, color='k')

plt.axis([0, 4, -0.1, 1.5])
#plt.axvline(0, color='k')
#plt.axhline(0, color='k')
plt.grid()
plt.legend()
plt.show()
plt.savefig('ex1.pdf', bbox_inches='tight')