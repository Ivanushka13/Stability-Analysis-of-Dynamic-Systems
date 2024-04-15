import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import cmath

from scipy.integrate import solve_ivp


class Binary_Telegraph_Process:
    """
    Инициализация телеграфного процесса
    :param size: размер выборки
    :param f: функция ошибки
    :param p1: матожидание ошибки в случае np.random.normal, low в случае np.random.uniform, alpha в случае np.random.beta
    :param p2: дисперсия ошибки в случае np.random.normal, high в случае np.random.uniform, beta в случае np.random.beta
    """
    
    def __init__(self, size, f, p1, p2=None, p3=None, RANDOM_SEED=123, alpha=1):
        self.size = size
        self.f    = f
        self.p1   = p1
        self.p2   = p2
        self.p3   = p3
        self.alpha = alpha
        self.RANDOM_SEED   = RANDOM_SEED
        self.synth_cps = []
        self.data = None
    
    def errors(self):
        if self.p1 == 'ECG':
            dir_ = 'ecg_data/sample_'+str(self.RANDOM_SEED)+'.csv'
            return pd.read_csv(dir_).values
        elif self.p1 == 'Si':
            dir_ = 'si_ar.csv'
            err = pd.read_csv(dir_)
            err = err[self.p2].dropna().values
            return err
        else:
            if self.p2 == None:
                return self.f(self.p1, size=self.size)
            elif self.p3 == None:
                return self.f(self.p1, self.p2, self.size)
            else:
                return self.f(self.p1, self.p2, self.p3, self.size)

    def euler(self, complex_mat):
        complex_num = np.linalg.det(complex_mat)

        r = np.abs(complex_num)
        phi = np.angle(complex_num)

        return r * (np.cos(phi) + 1j * np.sin(phi))


    def get_data(self):
        def f(omega, u):
            return 1j/2 * np.dot(omega, u)

        np.random.seed(self.RANDOM_SEED)
        data = []
        sigma_1 = np.array([[0, 1], [1, 0]])
        sigma_2 = np.array([[0, -1j], [1j, 0]])
        sigma_3 = np.array([[1, 0], [0, -1]])
        omega_1 = 1
        omega_2 = 2
        omega_3 = 3
        omega = omega_1 * sigma_1 + omega_2 * sigma_2 + omega_3 * sigma_3
        i = 0
        u_t = np.eye(2, dtype=complex)  # Начальное значение
        data.append(self.euler(u_t))
        errors = self.errors()
        errors -= np.mean(errors)
        errors /= np.std(errors)
        while i < self.size:
            u_t = u_t + f(omega, u_t) + self.alpha * errors[i]
            data.append(self.euler(u_t))
            i += 1
        self.data = data
        return data

    def data_to_eq(self):
        data = pd.DataFrame()
        data['solution'] = self.get_data()
        levels = []  # Разметка данных: 1 если был переход, 0 если не было
        levels.append(0)  # Первый элемент не с чем сравнивать, по умолчанию без перехода
        solve = self.euler([[0.99003812 - 0.11289082j, -0.07526055 - 0.03763027j],
                 [0.07526055 - 0.03763027j, 0.99003812 + 0.11289082j]])
        for i in range(1, len(self.data)):
            diff = data['solution'][i] - levels[i - 1]
            if np.all(np.abs(diff) > np.abs(solve)):
                if diff.all() > 0:
                    levels.append(levels[i - 1] + diff)
                else:
                    levels.append(levels[i - 1] - diff)
            else:
                levels.append(levels[i - 1])

        data['levels'] = levels

        return data


    def labels(self):
        labels = [0]
        data = self.data_to_eq()
        for i in range(1, data.shape[0]):
            if not np.allclose(data.iloc[i][1], data.iloc[i - 1][1]):
                labels.append(1)
            else:
                labels.append(0)

        self.labels_ = labels
        data['MARKS'] = labels

        return data
    
    def print_data(self, title='', width=50):
        fig = plt.figure(figsize=(30,10)) #визуализация
        font = {'size': 20}
        plt.rc('font', **font)
        table = self.labels()
        table['index'] = table.index
        for i in table['index']:
            x = table.iloc[i][3]
            y = table.iloc[i][0]
            level = table.iloc[i][1]
            mark = table.iloc[i][2]
            if mark == 1:
                if level == 0:
                    plt.scatter(x, y, marker = 'v', c = 'b', s=width)
                elif level == math.pi or level == -math.pi:
                    plt.scatter(x, y, marker = 'v', c = 'g', s=width)
                elif level == 2*math.pi or level == -2*math.pi:
                    plt.scatter(x, y, marker = 'v', c = 'y', s=width)
                else:
                    plt.scatter(x, y, marker = 'v', c = 'r', s=width)
            elif mark == 0:
                if level == 0:
                    plt.scatter(x, y, marker = 'o', c = 'b', s=width)
                elif level == math.pi or level == -math.pi:
                    plt.scatter(x, y, marker = 'o', c = 'g', s=width)
                elif level == 2*math.pi or level == -2*math.pi:
                    plt.scatter(x, y, marker = 'o', c = 'y', s=width)
                else:
                    plt.scatter(x, y, marker = 'o', c = 'r', s=width)
        plt.title(title, fontweight='bold')
        plt.xlabel('Time', fontweight='bold')
        plt.ylabel('Value', fontweight='bold')
        plt.grid()
        return fig