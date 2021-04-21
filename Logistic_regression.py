import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import math


class Logistic():
    def __init__(self, init):
        self.data = self.load_dataset()  # загрузка датасета
        self.x_train, self.t_train, self.x_valid, self.t_valid = self.dataset_split(self.data)  # делим датасет
        if int(init) == 1:
            self.w = np.random.normal(0, 0.1, (10, 64))  # норм распр-е
        elif int(init) == 2:
            self.w = (1 / self.x_train.shape[1]) * np.random.randn(10, 64) + 0  # xavier распр-е
        elif int(init) == 3:
            self.w = 1 * np.random.randn(10, 64) + 0  # He распр-е
        elif int(init) == 4:
            self.w = (4/12)*np.random.randn(10, 64) + 0  # равномерн распр-е ([-1, 1]); 4 = (1-(-1))^2
        self.w_best = self.w
        self.b_train = np.random.normal(0, 0.1, (10, len(self.x_train)))  # инициал-я b
        self.b_valid = np.random.normal(0, 0.1, (10, len(self.x_valid)))
        self.lr = 0.0001  # learning rate
        self.lambda_cur = 0.00001  # коэф регуляр-и
        self.e_min = 10 ** 10
        self.I = []
        self.E = []
        self.Acc_train, self.Acc_valid = [], []  # ошибки на train, valid
        self.batch_size = 32
        self.x_train_batch, self.target_train_batch, self.b_train_batch = [], [], []  # выборка, деленная на батчи
        self.split_on_batch()  # деление на батчи
        self.learning()  # запуск обучения

    def load_dataset(self): # загрузка данных
        return load_digits()

    def split_on_batch(self):
        size = math.ceil(self.x_train.shape[0] / self.batch_size)  # число батчей
        for i in range(size):
            self.x_train_batch.append(self.x_train[self.batch_size*i : self.batch_size*i + self.batch_size])
            self.target_train_batch.append(self.t_train[self.batch_size*i : self.batch_size*i + self.batch_size])
            self.b_train_batch.append(self.b_train[:, self.batch_size*i : self.batch_size*i + self.batch_size])

    def dataset_split(self, data):  # деление датасета
        target = self.hot_encoding(data.target)  # в one-hot представление
        n = data.target.shape[0]  # длина набора данных
        ind_s = np.arange(n)
        np.random.shuffle(ind_s)

        x = self.standart_data(data)  # стандартизация

        ind_train = ind_s[: np.int32(0.8 * n)]
        x_train = x[ind_train, :]
        t_train = target[ind_train, :]

        ind_valid = ind_s[np.int32(0.8 * n):]
        x_valid = x[ind_valid, :]
        t_valid = target[ind_valid, :]

        return x_train, t_train, x_valid, t_valid

    def standart_data(self, data):  # стандарт-я
        Mean = []
        Std = []
        for i in range(data.data.shape[1]):
            p = 0
            mean = data.data[:, i].sum()
            mean = mean / data.data.shape[0]  # среднее по всем компонентам
            variances = np.std(data.data[:, i], axis=0)  # стандарт. отклонение
            Mean.append(mean)
            Std.append(variances)
            mask = []
            for i in Std:
                if i > 0:
                    mask.append(True)  # если стандарт. отклонение > 0
                else:
                    mask.append(False)
            for i in range(len(mask)):
                if mask[i] == True:
                    data.data[:, i] = (data.data[:, i] - Mean[i]) / Std[i]  # стандартизация
            return data.data

    def hot_encoding(self, target):
        targets = []
        for i in target:
            t = np.zeros((10,), dtype=int)
            t[i] = 1  # единица на месте нужного класса
            targets.append(t)
        return np.asarray(targets)

    def softmax(self, w, x, b):
        a = (w.dot(x.T)) + b  # выход классификатора
        summ = []
        for i in range(a.shape[1]):
            summ.append(np.exp(a[:, i]).sum())  # сумма exp() по всей выборке
        Y = []
        for i in range(len(summ)):
            y = []
            max_a = -max(a[:, i])  # макс значение каждого элемента а
            for j in range(a.shape[0]):
                y.append(np.exp(a[j, i]) / summ[i])
            Y.append(y)
        return Y  # выход softmax-а

    def get_E(self, t, y, x):  # подсчет ошибки
        e = 0
        y = np.array(y)
        for i in range(x.shape[0]):  # по всем элементам выборки
            for k in range(10):
                e += t[i, k] * np.log(y[i, k])
        return e

    def gradient(self, w, y, t, x):
        grad = ((y - t).T).dot(x) + self.lambda_cur * w
        return grad

    def show(self):  # графики точности на трен и валид выборках
        plt.figure()
        plt.plot(self.I, self.E, '-', color='darkorange')
        plt.show()
        plt.plot(self.I, self.Acc_train, '--', color='blue')
        plt.plot(self.I, self.Acc_valid, '--', color='green')
        plt.legend(("acc_train", "acc_valid"))
        plt.show()

    def matrix(self, y, t):  # confusion matrix
        matrix = np.zeros((10, 10), dtype=int)  # подсчет точности, уверенные правильные и уверенные неправильные
        for z in range(t.shape[0]):
            max_y = y[z].index(max(y[z]))
            max_t = np.argmax(t[z])
            matrix[max_t, max_y] += 1
        return matrix

    def accuracy(self, matrix):  # подсчет accuracy
        acc_valid, v = 0, 0
        for z in range(10):
            for r in range(10):
                if z == r:
                    acc_valid += matrix[z, z]
                v += matrix[z, r]
        return acc_valid / v

    def learning(self):
        y_v = self.softmax(self.w, self.x_valid, self.b_valid)

        matrix_valid = self.matrix(y_v, self.t_valid)  # составление confusion matrix до обучения
        acc_valid = self.accuracy(matrix_valid)   # подсчет accuracy до обучения

        print('Accuracy on valid before learning', acc_valid*100)

        # сonfusion matrix до обучения
        plt.figure(figsize=(9, 9))
        plt.imshow(matrix_valid, interpolation='nearest', cmap='Blues')
        plt.title('Confusion matrix before', size=15)
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
        plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
        plt.tight_layout()
        plt.ylabel('Actual label', size=15)
        plt.xlabel('Predicted label', size=15)
        width, height = matrix_valid.shape
        for x in range(width):
            for y in range(height):
                plt.annotate(str(matrix_valid[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')

        # обучение
        for i in range(120):
            for packet, target, b in (zip(self.x_train_batch, self.target_train_batch, self.b_train_batch)):  # проход по батчам
                y = self.softmax(self.w, packet, b)
                e = -self.get_E(target, y, packet)
                grad = self.gradient(self.w, y, target, packet)
                w_new = self.w - self.lr * grad
                self.I.append(i)
                self.E.append(e)
                y_v = self.softmax(w_new, self.x_valid, self.b_valid)
                e_cur = -self.get_E(self.t_valid, y_v, self.x_valid)
                if e_cur < self.e_min:  # валидация
                    self.e_min = e_cur
                    self.w_best = w_new
                matrix_train = self.matrix(y, target)  # составление confusion matrix
                matrix_valid = self.matrix(y_v, self.t_valid)
                acc_train = self.accuracy(matrix_train) * 100  # подсчет accuracy
                acc_valid = self.accuracy(matrix_valid) * 100
                self.Acc_train.append(acc_train)
                self.Acc_valid.append(acc_valid)
                if (i + 1) % 30 == 0:  # вывод ошибок
                    print('e-train', e, 'e-val', e_cur)
                    print('acc-tr', acc_train, 'acc-val', acc_valid)

                if np.linalg.norm(w_new - self.w) < 10 ** (-3):  # разница последовательных приближений
                    break
                if np.linalg.norm(grad) < 10 ** (-3):  # норма градиента
                    break
                self.w = w_new

        # точность на валид-йо, вывод топ-правильно и неправильно класифицированных чисел
        y_v = self.softmax(self.w_best, self.x_valid, self.b_valid)
        matrix_valid = self.matrix(y_v, self.t_valid)
        acc_valid, v = 0, 0
        tp, TP = [-1, -1, -1], [-1, -1, -1]  # TP
        fp, FP = [-1, -1, -1], [-1, -1, -1]  # FP
        for z in range(10):
            for r in range(10):
                if z == r:
                    acc_valid += matrix_valid[z, z]
                    a = tp.index(min(tp))  # минимальное значение уверенности из правильно-классифицированных
                    if matrix_valid[z, z] > tp[a]:
                        tp[a] = matrix_valid[z, z]
                        TP[a] = z
                else:
                    a = fp.index(min(fp))  # минимальное значение уверенности из неправильно-классифицированных
                    if matrix_valid[z, r] > fp[a]:
                        fp[a] = matrix_valid[z, r]
                        FP[a] = r
                v += matrix_valid[z, r]
        acc_valid /= v

        print('Accuracy on valid after learning', acc_valid*100)

        self.show()  # графики точности на трен и валид выборках

        # confusion matrix после обучения
        plt.figure(figsize=(9, 9))
        plt.imshow(matrix_valid, interpolation='nearest', cmap='Blues')  # Pastel2
        plt.title('Confusion matrix', size=15)
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
        plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
        plt.tight_layout()
        plt.ylabel('Actual label', size=15)
        plt.xlabel('Predicted label', size=15)
        width, height = matrix_valid.shape
        for x in range(width):
            for y in range(height):
                plt.annotate(str(matrix_valid[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')

        plt.figure(figsize=(18, 6))  # вывод уверенно-правильных предсказаний
        for index, (image, label) in enumerate(zip(self.data.data[TP[:]],
                                                   self.data.target[TP[:]])):
            plt.subplot(1, 6, index + 1)
            plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
            plt.title('TOP-TP: %i\n' % label, fontsize=20)
        plt.show()
        plt.figure(figsize=(18, 6))  # вывод уверенно-неправильных предсказаний
        for index, (image, label) in enumerate(zip(self.data.data[FP[:]],
                                                   self.data.target[FP[:]])):
            plt.subplot(1, 6, index + 1)
            plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
            plt.title('TOP-FP: %i\n' % label, fontsize=20)
        plt.show()


init = input("Введите цифру (выбор начальной иниц-ии): 1-норм-ое, 2-Xavier, 3-He, 4-равн-ое \n ")
model = Logistic(init)
