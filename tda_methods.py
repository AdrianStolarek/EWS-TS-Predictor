import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from gtda.time_series import SlidingWindow
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape

class StockPricePredictor:
    def __init__(self, file_path, time_step=100):
        self.file_path = file_path
        self.time_step = time_step
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        data = pd.read_csv(self.file_path, parse_dates=['Date'])
        data.set_index('Date', inplace=True)
        self.data = data
        self.scaled_data = self.scaler.fit_transform(data[['Co2']])

    def preprocessed_data(self):
        X, Y = [], []
        for i in range(len(self.scaled_data) - self.time_step-1):
            a = self.scaled_data[i:(i+self.time_step), 0]
            X.append(a)
            Y.append(self.scaled_data[i + self.time_step, 0])
        self.X, self.Y = np.array(X), np.array(Y)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)

    def train_test_split(self, test_size=0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(532, return_sequences=True, input_shape = (self.time_step, 1)))
        self.model.add(LSTM(700, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

    def compile_and_train(self, epochs=30, batch_size=1, patience=10):
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model.fit(self.X_train, self.Y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, callbacks=[early_stop])

    def evaluate_model(self):
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)

        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)

        train_rmse = np.sqrt(np.mean((train_predict - self.scaler.inverse_transform(self.Y_train.reshape(-1,1)))**2))
        test_rmse = np.sqrt(np.mean((test_predict - self.scaler.inverse_transform(self.Y_test.reshape(-1,1)))**2))

        print(f'Train RMSE: {train_rmse}')
        print(f'Test RMSE: {test_rmse}')

    def make_predictions(self):
        plt.figure(figsize=(14,5))
        plt.plot(self.data.index, self.scaler.inverse_transform(self.scaled_data), label='True Data')

        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)

        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)

        train_len = len(self.X_train)
        test_len = len(self.X_test)

        plt.plot(self.data.index[self.time_step:train_len + self.time_step], train_predict, label='Train Prediction')
        plt.plot(self.data.index[train_len + 2*self.time_step:], test_predict, label='Test Prediction')

        plt.xlabel('Date')
        plt.ylabel('NASDAQ Index')
        plt.legend()
        plt.show()


predictor = StockPricePredictor(file_path="US_NASDAQ.csv", time_step=100)
predictor.load_data()
predictor.preprocessed_data()
predictor.train_test_split()
predictor.build_model()
predictor.compile_and_train(epochs=20, batch_size=1)
predictor.evaluate_model()
predictor.make_predictions()