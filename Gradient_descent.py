import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

N = 1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

lambda_cur = 10**(-8)


def gradient(w, F, t):
    grad = ((w.T).dot(F.T)). dot(F) - t.T.dot(F)
    return grad


def dataset_split():
    ind_s = np.arange(N)
    np.random.shuffle(ind_s)
    ind_train = ind_s[: np.int32(0.8*N)]
    x_train = x[ind_train]
    t_train = t[ind_train]

    ind_valid = ind_s[np.int32(0.8*N): np.int32(0.9 * N)]
    x_valid = x[ind_valid]
    t_valid = t[ind_valid]

    ind_test = ind_s[np.int32(0.9*N):]
    x_test = x[ind_test]
    t_test = t[ind_test]

    return x_train, t_train, x_valid, t_valid, x_test, t_test


def get_F(count):
    F = np.ones((len(x_train), count + 1), dtype=float)  # матрица (len(x_train) х (число ф-ий + 1))
    for i in range(1, count + 1):
        p = np.random.randint(1, 13, size=len(x_train))
        F[:, i] = x_train**p
    return F


def get_param(F, count):
    I = np.eye(count + 1)  # единичная матрица
    w = ((np.linalg.inv(F.T.dot(F) + lambda_cur*I)).dot(F.T)).dot(t_train)
    return w


def standart(a):
    mean = np.mean(a, axis=0)
    varian = np.std(a, axis=0)
    x = (a - mean)/varian
    return x


x_train, t_train, x_valid, t_valid, x_test, t_test = dataset_split()


count = np.random.randint(2, 15)
F = get_F(count)
w = get_param(F, count)
gamma = 0.01

for i in range(N):
    print(F.shape, w.shape)
    gr = gradient(w, F, t_train)
    w_new = w - gamma*gr
    norm = np.linalg.norm(gr)
    if norm < 10**(-3):
        break
    w = w_new


print(w)

