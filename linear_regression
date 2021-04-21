import numpy as np
import matplotlib.pyplot as plt
import random

N = 1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)   #случ ошибка
t = z + error

# prediction
def LR(M):
    F = np.zeros((N, M + 1), dtype=float)
    for i in range(M + 1):
        F[:, i] = x ** i
    w = ((np.linalg.inv(F.T.dot(F))).dot(F.T)).dot(t)
    y = w.dot(F.T) # F.dot(w)
    return y


# степень M
M = [1, 8, 100]

# вывод графиков
for m in M:
    y = LR(m)
    plt.figure()
    plt.title('M={}'.format(m))
    plt.plot(x, t, 'b. ')
    plt.plot(x, z, 'r.-')
    plt.plot(x, y, 'g.-')
    plt.show()

M = []  # степени
E = []  # ошибки


# подсчет ошибок
for m in range(1, 101):
    y = LR(m)
    e = 0
    for k in range(N):
        e += (y[k] - t[k]) ** 2
    E.append(e/2)
    M.append(m)


# зависимость ошибки от степени
plt.figure()
plt.title('ERRORS')
plt.plot(M, E, 'k.-')
plt.show()



