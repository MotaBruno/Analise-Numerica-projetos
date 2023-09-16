import numpy
import matplotlib.pyplot as plt

def f(x):
    return x**2 + (np.sin(9*x))**2

n=8
xi = [i/n for i in range(n+1)]
yi = [f(xi[i]) for i in range(n+1)]
h = 1/n

#sistema para encontrar os Ms:
A = np.array([[h/6,2*h/3,h/6,0,0,0,0,0,0], [0,h/6,2*h/3,h/6,0,0,0,0,0], [0,0,h/6,2*h/3,h/6,0,0,0,0],[0,0,0,h/6,2*h/3,h/6,0,0,0],[0,0,0,0,h/6,2*h/3,h/6,0,0],[0,0,0,0,0,h/6,2*h/3,h/6,0],[0,0,0,0,0,0,h/6,2*h/3,h/6],[1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]])
B1 = [(yi[i+1]-yi[i])/h - (yi[i]-yi[i-1])/h for i in range(1,n)]+[0,0]
B = np.array(B1)
M = np.linalg.solve(A,B)

#função spline:
def s(x):
    if 0<=x<xi[1]:
        return 1/(6*h)*M[1]*(x-xi[0])**3 + yi[0]*(xi[1]-x)/h + (yi[1]-(M[1]*h**2)/6)*(x-xi[0])/h

    if x<xi[2]:
        return 1/(6*h)*M[1]*(xi[2]-x)**3 + 1/(6*h)*M[2]*(x-xi[1])**3 + (yi[1]-(M[1]*h**2)/6)*(xi[2]-x)/h + (yi[2]-(M[2]*h**2)/6)*(x-xi[1])/h

    if x<xi[3]:
        return 1/(6*h)*M[2]*(xi[3]-x)**3 + 1/(6*h)*M[3]*(x-xi[2])**3 + (yi[2]-(M[2]*h**2)/6)*(xi[3]-x)/h + (yi[3]-(M[3]*h**2)/6)*(x-xi[2])/h

    if x<xi[4]:
        return 1/(6*h)*M[3]*(xi[4]-x)**3 + 1/(6*h)*M[4]*(x-xi[3])**3 + (yi[3]-(M[3]*h**2)/6)*(xi[4]-x)/h + (yi[4]-(M[4]*h**2)/6)*(x-xi[3])/h

    if x<xi[5]:
        return 1/(6*h)*M[4]*(xi[5]-x)**3 + 1/(6*h)*M[5]*(x-xi[4])**3 + (yi[4]-(M[4]*h**2)/6)*(xi[5]-x)/h + (yi[5]-(M[5]*h**2)/6)*(x-xi[4])/h

    if x<xi[6]:
        return 1/(6*h)*M[5]*(xi[6]-x)**3 + 1/(6*h)*M[6]*(x-xi[5])**3 + (yi[5]-(M[5]*h**2)/6)*(xi[6]-x)/h + (yi[6]-(M[6]*h**2)/6)*(x-xi[5])/h

    if x<xi[7]:
        return 1/(6*h)*M[6]*(xi[7]-x)**3 + 1/(6*h)*M[7]*(x-xi[6])**3 + (yi[6]-(M[6]*h**2)/6)*(xi[7]-x)/h + (yi[7]-(M[7]*h**2)/6)*(x-xi[6])/h

    if x<=1:
        return 1/(6*h)*M[7]*(xi[8]-x)**3 + (yi[7]-(M[7]*h**2)/6)*(xi[8]-x)/h + yi[8]*(x-xi[7])/h

S=np.vectorize(s)

#gráficos:
x = np.arange  ( start = 0
                , stop = 1.001
                , step = 0.001
                )

y = S(x)
ff = f(x)
plt.plot(x,ff, label = 'f(x)')
plt.plot(x, y, 'r-', label = 's(x)')


pontos = np.array([ [xi[i],yi[i]] for i in range(n+1) ])
xx, yy = pontos.T
plt.scatter(xx,yy, color='k')

plt.axis([0, 1, -0.5, 1.8])
#plt.axvline(0, color='k')
#plt.axhline(0, color='k')
plt.grid()
plt.legend()
plt.show()
plt.savefig('ex2-spline.pdf', bbox_inches='tight')
