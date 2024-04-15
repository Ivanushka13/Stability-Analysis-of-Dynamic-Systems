import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import accuracy_score #здесь accuracy - плохая метрика, потому что классы несбалансированные
from sklearn.metrics import f1_score #f1 рассчитывается для класса 1
from sklearn.metrics import auc, precision_recall_curve
from sklearn.mixture import GaussianMixture


class binary_model:
    """
    LSTM модель, предсказывающая переход на новый уровень
    :param data: выборка, list
    :param percent: сколько процентов выборки составит train, float
    :param window: размер батча, int
    """
    def __init__(self, data, percent, window=10, model='gmm', n_epochs=100):
        self.data = data
        self.percent = percent
        self.window = window
        self.model = model
        self.models_ = []
        self.n_epochs = n_epochs
        
    def train_test_split(self):
        end = round(len(self.data)*self.percent)
        start = end - self.window
        Y_train = self.data[0:end]
        Y_train = [[i] for i in Y_train]
        Y_train = np.array(Y_train)
        Y_test = self.data[start:]
        return Y_train, Y_test
    
    def data_reshape(self, sample): #используется только внутри класса, не вызывается
        features_set = [] #хранит массив 1*10 первых чисел
        levels = [] #хранит 11-е "предсказанные" числа
        for i in range(self.window, sample.shape[0]):
            features_set.append(sample[i-self.window:i, 0])#начинает идти с 0, потом с 1, потом с 2 и тд до i невключительно
            levels.append(sample[i, 0]) #добавляет каждое следующее i-е число
        features_set, levels = np.array(features_set), np.array(levels)
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
        return features_set, levels
    
    def model_fit(self, verbose=0):
        np.random.seed(1)
        tf.random.set_seed(2)
        Y_train = self.train_test_split()[0]
        sample, answers = self.data_reshape(Y_train)
        
        model = Sequential()
        model.add(LSTM(units=5, return_sequences=True, input_shape=(sample.shape[1], 1), use_bias=False))
        model.add(Dropout(0.2))

        model.add(LSTM(units=5, use_bias=False))
        model.add(Dropout(0.2))

        model.add(Dense(16))
        model.add(Dropout(0.2))

        model.add(Dense(1))
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'mean_squared_error')

        model.fit(sample, answers, epochs = self.n_epochs, batch_size = 32, verbose=verbose)
        
        self.models_.append(model)
        
        return self
    
    def model_predict(self, how_many):
        Y_test = self.train_test_split()[1]
        if how_many == 1:
            Y_test = [[i] for i in Y_test]
            Y_test = np.array(Y_test)
            new_features_set = self.data_reshape(Y_test)[0]
            
            predictions = self.models_[0].predict(new_features_set, verbose=0).flatten().tolist()
            
        else:
            predictions = []
            start = 0
            end = len(self.data) - round(len(self.data)*self.percent)
            while len(predictions) < end:
                sample = list(Y_test[start:start+self.window])
                for i in range(how_many):
                    scaled_sample = [[[i] for i in sample]]
                    scaled_sample = np.array(scaled_sample)#tf.convert_to_tensor(, dtype=tf.float64)
                    new = self.models_[0].predict(scaled_sample, verbose=0).tolist()[0][0]
                    predictions.append(new)
                    sample.append(new)
                    sample = sample[1:]
                start = start + how_many
        return predictions
    
    def prediction_show(self, predictions):
        end = round(len(self.data)*self.percent)
        fig = plt.figure(figsize=(30,10)) #визуализация
        initial = self.data[end:]  
        x = list(range(0, len(initial)))
        plt.scatter(x, initial)
        plt.scatter(x, predictions)
        plt.axhline(y = math.pi, color = 'r')
        plt.axhline(y = -math.pi, color = 'r')
        plt.axhline(y = 0)
        return plt.show()

    def euler(self, complex_mat):
        complex_num = np.linalg.det(complex_mat)

        r = np.abs(complex_num)
        phi = np.angle(complex_num)

        return r * (np.cos(phi) + 1j * np.sin(phi))
    
    def data_to_eq(self, predictions):
        data = pd.DataFrame()
        levels = []  # Разметка данных: 1 если был переход, 0 если не было
        levels.append(0)  # Первый элемент не с чем сравнивать, по умолчанию без перехода
        solve = self.euler([[0.99003812 - 0.11289082j, -0.07526055 - 0.03763027j],
                            [0.07526055 - 0.03763027j, 0.99003812 + 0.11289082j]])
        for i in range(1, len(self.data)):
            diff = predictions[i] - levels[i - 1]
            if np.all(np.abs(diff) > np.abs(solve)):
                if diff.all() > 0:
                    levels.append(levels[i - 1] + diff)
                else:
                    levels.append(levels[i - 1] - diff)
            else:
                levels.append(levels[i - 1])

        data['levels'] = levels
        data['data_eq'] = predictions
        return data
    
    def labels(self, predictions):
        labels = [0]
        data = self.data_to_eq(predictions)
        for i in range(1, data.shape[0]):
            if not np.allclose(data.iloc[i][1], data.iloc[i - 1][1]):
                labels.append(1)
            else:
                labels.append(0)

        self.labels_ = labels
        data['MARKS'] = labels

        return data
    
    def metrics(self, y_true, y_pred): #real_marks - dataframe
        # if y_true == None:
        #     y_true = y_true['MARKS'].tolist() #marks - столбец dataframe из нулей и единиц
        y_true = y_true[len(y_true)-len(y_pred):]
        y_true = y_true[1:]
        y_pred = y_pred[1:]
        precision, recall, thr = precision_recall_curve(y_true, y_pred)
        
        f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
        acc = accuracy_score(y_true, y_pred)
        pr_auc = auc(recall, precision)
        return f1, acc, pr_auc