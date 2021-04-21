import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


# вычисление градиента
def gradient(w, F, t, count, lambda_cur):
    I = np.eye(count)
    grad = w.T.dot((F.T).dot(F)+lambda_cur*I) - t.T.dot(F)
    return grad


# деление датасета
def dataset_split(data):
    n = data.target.shape[0]  # длина набора данных
    x = standart_data(data)

    x_train = x[: np.int32(0.8 * n), :]
    t_train = data.target[: np.int32(0.8 * n)]

    x_test = x[np.int32(0.8*n):, :]
    t_test = data.target[np.int32(0.8*n):]

    return x_train, t_train, x_test, t_test


def get_F(x, count, numb):
    F = np.ones((len(x), numb), dtype=float)
    for i in range(1, count+1):
        F[:, i] = x[:, i-1]
    for i in range(count + 1, numb):
        y = 1
        p = np.random.randint(0, 1, size=count)  # степени
        for k in range(count):
            y = y * (x[:, k]**p[k])
        F[:, i] = y
    return F


def standart_data(d):
    for i in range(d.data.shape[1]):
        mean = np.mean(d.data[:, i], axis=0)
        variances = np.std(d.data[:, i], axis=0)
        d.data[:, i] = (d.data[:, i] - mean)/variances
    return d.data


def get_error(f, w, x, t, lambda_cur):
    y = f.dot(w)
    e = 0
    for k in range(len(x)):
        e += (y[k] - t[k]) ** 2
    e = e * 1/2 + lambda_cur*1/2 * (w.T.dot(w))
    return e


data = load_boston()
x_train, t_train, x_test, t_test = dataset_split(data)  # делим датасет
t = data.target

lambda_cur = 10**(-7)  # коэф регуляризации
lr = 0.000001  # learning rate
count = data.data.shape[1]
number = np.random.randint(6, 20)  # (13+number) - число столбцов в матрице плана
w = np.random.normal(0, 0.1, count + number)  # начальная инициализация пар-ов модели
F = get_F(x_train, count, count + number)  # матрица плана

E = []  # ошибки
I = []  # итерации

for i in range(4000):
    gr = gradient(w, F, t_train, count + number, lambda_cur)
    w_new = w - lr*gr
    norm = np.linalg.norm(gr)
    er = get_error(F, w, x_train, t_train, lambda_cur)
    E.append(er)
    I.append(i)
    if norm < 10**(-3):
        break
    if np.linalg.norm(w - w_new) < 10 ** (-3):
        break
    w = w_new

plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(I, E, 'b-')
plt.show()

f = get_F(x_test, count, number + count)
E_train = get_error(F, w, x_train, t_train, lambda_cur)
E_model = get_error(f, w, x_test, t_test, lambda_cur)
print("E_train: ", E_train)
print("E_test: ", E_model)

