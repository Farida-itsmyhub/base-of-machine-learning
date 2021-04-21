import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

max_depth = 20
x = 7  # число интервалов
s_true = [0 for i in range(7)]  # правильно-классифицированные
s_false = [0 for i in range(7)]  # ошибочно-классифицированные

matrix_train = np.zeros((10, 10), dtype=int)  # conf matrix для train
matrix_test = np.zeros((10, 10), dtype=int)  # conf matrix для train


def dataset_split(data):
    target = hot_encoding(data.target)  # в one-hot представление
    n = data.target.shape[0]  # длина набора данных
    ind_s = np.arange(n)
    np.random.shuffle(ind_s)
    ind_train = ind_s[: np.int32(0.8 * n)]

    x_train = data.data[ind_train, :]
    t_train = target[ind_train, :]

    ind_valid = ind_s[np.int32(0.8 * n):]
    x_valid = data.data[ind_valid, :]
    t_valid = target[ind_valid, :]

    return x_train, t_train, x_valid, t_valid


def hot_encoding(target):
    targets = []
    for i in target:
        t = np.zeros((10,), dtype=int)
        t[i] = 1
        targets.append(t)
    return np.asarray(targets)


def get_params(data):
    par = np.random.choice(data.shape[1], size=data.shape[1], replace=False)
    return par


def create_split_node(p, t):
    Node = {}
    Node['threshold'] = t
    Node['coord_int'] = p
    Node['is_terminal'] = False
    return Node


def Stop(data, t, depth):
    if depth >= max_depth:
        return True
    # количество элементов выборки
    if data.shape[0] > 60:
        return False
    #  вычисление энтропии
    h_i = 0
    n_i = len(data)
    n_i_k = np.zeros((10,), dtype=int)
    for i in range(data.shape[0]):
        l = np.argmax(t[i])
        n_i_k[l] += 1
    for i in range(10):
        if (n_i_k[i]) != 0 and n_i != 0:
            h_i += (n_i_k[i] / n_i) * (np.log(n_i_k[i]) - np.log(n_i))
    if ((-1) * h_i) < 0.15:
        return True
    return True


def create_terminal_node(data, t):
    Node = {}
    Node['is_terminal'] = True
    N_i = len(data)
    N_i_k = np.zeros((10,), dtype=int)
    for i in range(len(data)):
        l = np.argmax(t[i])
        N_i_k[l] += 1
    if (N_i==0):
        Node['vector'] = N_i_k
        Node['components'] = N_i_k
        return Node
    Node['vector'] = N_i_k / N_i
    Node['components'] = N_i_k
    return Node


def split_data(data, t, par):
    # прирост энтропии
    T, H, entropy = [], [], []  # массивы с tau,inform gain
    for p in par:
        n = np.unique(data[:, p])
        tau = n[0]
        I, e = [], []
        for k in n:  # перебор по границам
            left, t_l, right, t_r = [], [], [], []  # вектора хар-к и соответствующие им target
            for i in range(data.shape[0]):
                if data[i, p] < k:
                    left.append(data[i])
                    t_l.append(t[i])
                else:
                    right.append(data[i])
                    t_r.append(t[i])
            #  вычисление энтропии
            H_i = 0
            N_i = len(data)
            N_i_k = np.zeros((10,), dtype=int)
            for i in range(data.shape[0]):
                l = np.argmax(t[i])  # data.data[i].index(max(data.data[i])) # индекс макс эл-та
                N_i_k[l] += 1
            for i in range(10):
                if (N_i_k[i]) != 0 and N_i != 0:
                    H_i += (N_i_k[i] / N_i) * (np.log(N_i_k[i]) - np.log(N_i))
            H_i = (-1) * H_i
            H_i_j, H_i_j_1, H_i_j_2 = 0, 0, 0
            N_i_j_1, N_i_j_2 = np.zeros((10,), dtype=int), np.zeros((10,), dtype=int)  # [], []

            for i in range(len(left)):
                l = np.argmax(t_l[i])  # индекс макс эл-та
                N_i_j_1[l] += 1
            for i in range(10):
                if (N_i_j_1[i]) != 0 and N_i != 0:
                    H_i_j_1 += (N_i_j_1[i] / len(left)) * (np.log(N_i_j_1[i]) - np.log(len(left)))
            H_i_j_1 = (-1) * (H_i_j_1) * len(left) / N_i

            for i in range(len(right)):
                l = np.argmax(t_r[i])  # индекс макс эл-та
                N_i_j_2[l] += 1
            for i in range(10):
                if (N_i_j_2[i]) != 0 and N_i != 0:
                    H_i_j_2 += (N_i_j_2[i] / len(right)) * (np.log(N_i_j_2[i]) - np.log(len(right)))
            H_i_j_2 = (-H_i_j_2) * len(right) / N_i

            H_i_j = H_i_j_1 + H_i_j_2

            h = H_i - H_i_j

            I.append(h)  # inform gain для компоненты
            if (h) >= max(I):
                tau = k

        H.append(max(I))
        T.append(tau)

    index = par[np.argmax(H)]  # индекс, когда наибольший inform gain
    tau = T[np.argmax(H)]  # лучший tau

    left, t_l, right, t_r = [], [], [], []
    for i in range(data.shape[0]):
        if data[i, index] < tau:
            left.append(data[i])
            t_l.append(t[i])
        else:
            right.append(data[i])
            t_r.append(t[i])
    return np.array(left), np.array(right), t_l, t_r, tau, index


def create_Tree(data, target, depth):
    if not Stop(data, target, depth):
        params = get_params(data)
        left_data, right_data, t_l, t_r, tau, index = split_data(data, target, params)
        node = create_split_node(index, tau)
        node['left'] = create_Tree(left_data, t_l, depth + 1)
        node['right'] = create_Tree(right_data, t_r, depth + 1)
    else:
        node = create_terminal_node(data, target)
        # построение confusion matrix
        pred = np.argmax(node["vector"])
        t = node["components"]
        for z in range(10):
            matrix_train[z, pred] += t[z]
    return node


# разделение данных при тестировании
def delenie(node, data, t):
    left, t_l, right, t_r = [], [], [], []
    for i in range(data.shape[0]):
        if data[i, node["coord_int"]] < node["threshold"]:
            left.append(data[i])
            t_l.append(t[i])
        else:
            right.append(data[i])
            t_r.append(t[i])
    return np.array(left), np.array(right), t_l, t_r


#  test
def Model(node, data, target):
    if node['is_terminal'] == False:
        left_data, right_data, t_l, t_r = delenie(node, data, target)
        node['left'] = Model(node["left"], left_data, t_l)
        node['right'] = Model(node["right"], right_data, t_r)
    else:
        N_i = len(data)
        N_i_k = np.zeros((10,), dtype=int)
        for i in range(len(data)):
            l = np.argmax(target[i])
            N_i_k[l] += 1
        if N_i == 0:
            node['vector'] = N_i_k
            node['components'] = N_i_k
        else:
            node['vector'] = N_i_k / N_i
            node['components'] = N_i_k
            # построение confusion matrix
            pred = np.argmax(node["vector"])
            confidence = max(node["vector"])
            number = max(node["components"])
            sum = node['components'].sum()
            t = node["components"]
            i = int(confidence * 6)
            s_true[i] += number
            s_false[i] += (sum - number)
            for z in range(10):
                matrix_test[z, pred] += t[z]
    return node


data = load_digits()
x_train, t_train, x_valid, t_valid = dataset_split(data)  # делим датасет

model = create_Tree(x_train, t_train, 1)

print("CONFUSION MATRIX ON TRAIN")
print(matrix_train)
acc_valid, v = 0, 0
for z in range(10):
    for r in range(10):
        if z == r:
            acc_valid += matrix_train[z, z]
        v += matrix_train[z, r]
print("accuracy_train: ", acc_valid / v * 100)

y = Model(model, x_valid, t_valid)
print("CONFUSION MATRIX ON TEST")
print(matrix_test)
acc_test, t = 0, 0
for z in range(10):
    for r in range(10):
        if z == r:
            acc_test += matrix_test[z, z]
        t += matrix_test[z, r]
print("accuracy_test: ", acc_test / t * 100)



# правильно-классифицированные
ax = plt.gca()
ax.bar(range(x), s_true, align='edge')
plt.title('True class')
plt.show()
# ошибочно-классифицированные
ax = plt.gca()
ax.bar(range(x), s_false, align='edge')
plt.title('False class')
plt.show()

