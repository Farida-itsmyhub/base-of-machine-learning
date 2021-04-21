import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


N = 1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

# возможные лямбды
lambda_set = [0, 10**(-7), 10**(-9), 10**(-8), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 5, 10, 50, 1000]
# набор функций
fi_set = [np.sin, np.cos, np.tan, lambda x: x**2, lambda x: x**3, lambda x: x**5, lambda x: x**7, lambda x: x**9, lambda x: x**11, lambda x: x**13, lambda x: x**15, lambda x: x**14, lambda x: x**6, lambda x: x**8, lambda x: x**16, lambda x: x**20, lambda x: x**23, lambda x: np.cos(x)*np.sin(x), np.exp, np.sqrt]

# деление датасета в соотношении [80, 10, 10]
def dataset_split():
    ind_s = np.arange(N)
    np.random.shuffle(ind_s)
    ind_train = ind_s[: np.int32(0.8 * N)]
    x_train = x[ind_train]
    t_train = t[ind_train]

    ind_train = ind_s[np.int32(0.8 * N): np.int32(0.9 * N)]
    x_valid = x[np.int32(0.8 * N):np.int32(0.9 * N)]
    t_valid = t[np.int32(0.8 * N):np.int32(0.9 * N)]

    ind_test = ind_s[np.int32(0.9*N):]
    x_test = x[ind_test]
    t_test = t[ind_test]

    return x_train, t_train, x_valid, t_valid, x_test, t_test

# выбор коэф-та регуляризации
def get_lambda(lambda_set):
    lambda_cur = np.random.choice(lambda_set, 1, replace=False)
    return lambda_cur

# выбор функций фи
def get_fi(fi_set, count):
    fi_cur = np.random.choice(fi_set, count, replace=False)
    return fi_cur

# вычисление параметров w
def get_param(fi_cur, lambda_cur, x_train, t_train):
    I = np.eye(len(fi_cur)+1) # единичная матрица
    F = np.ones((len(x_train), (len(fi_cur) + 1)), dtype=float) # матрица (len(x_train) х (число ф-ий + 1))
    for i in range(1, len(fi_cur) + 1):
        F[:, i] = fi_cur[i - 1](x_train)
    w = ((np.linalg.inv(F.T.dot(F) + lambda_cur*I)).dot(F.T)).dot(t_train)
    return w

# вычисление ошибки
def get_error(fi_cur, lambda_cur, w, x, t):
    F = np.ones((len(x), len(fi_cur) + 1), dtype=float)
    for i in range(1, len(fi_cur)+1):
        F[:, i] = fi_cur[i-1](x)
    y = F.dot(w)
    e = 0
    for k in range(len(x)):
        e += (y[k] - t[k]) ** 2
    e = e * 1/2 + lambda_cur*1/2 * (w.T.dot(w))
    return e


x_train, t_train, x_valid, t_valid, x_test, t_test = dataset_split()

iter = 1000
E_min = 10**10
lambda_best = 0
w_best = 0
fi_best = 0

for i in range(iter):
    count = np.random.randint(1, len(fi_set))
    lambda_cur = get_lambda(lambda_set)
    fi_cur = get_fi(fi_set, count)
    w = get_param(fi_cur, lambda_cur, x_train, t_train)
    E_cur = get_error(fi_cur, lambda_cur, w, x_valid, t_valid)

    if E_cur < E_min:
        E_min = E_cur
        lambda_best = lambda_cur
        w_best = w
        fi_best = fi_cur

# ошибка на test-части
E_model = get_error(fi_best, lambda_best, w_best, x_test, t_test)
# вывод ошибки на test-части, лучшей лямбда, лучших фи
print(list(map('E = {:.6f}'.format, E_model)))
print(list(map('lambda = {:.12f}'.format, lambda_best)))
print(w_best.shape)
print('fi = {}'.format, fi_best)
F = np.ones((len(x), len(fi_best)+1), dtype=float)
for i in range(1, len(fi_best)+1):
    F[:, i] = fi_best[i - 1](x)

y = F.dot(w_best)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, t, 'b. ', alpha=0.4)  # график t(x)
plt.plot(x, z, 'r.-', alpha=0.23)  # график z(x)
ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, 40)))  # для красивого цвета :)
for n in range(27):  # график регрессии
    ax.plot(x, y, lw=3)
plt.show()
