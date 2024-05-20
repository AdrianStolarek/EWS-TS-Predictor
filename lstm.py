import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('DATA_ALL.csv')

df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')

df = df.sort_values('Data')

data = df['liczba_wszystkich_zakazen'].values

scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 14
X, Y = create_dataset(scaled_data, look_back)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=10, batch_size=1, verbose=2)

def predict_future(data, model, look_back, days):
    predictions = []
    current_batch = data[-look_back:]
    current_batch = current_batch.reshape((1, look_back, 1))

    for _ in range(days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred[0])
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    return predictions


future_days = 7
future_predictions = predict_future(scaled_data, model, look_back, future_days)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

future_dates = pd.date_range(df['Data'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
future_df = pd.DataFrame({'Data': future_dates, 'Predykcja': future_predictions.flatten()})

print(future_df)

plt.figure(figsize=(12, 6))
plt.plot(df['Data'], df['liczba_wszystkich_zakazen'], label='Historyczne dane')
plt.plot(future_df['Data'], future_df['Predykcja'], label='Prognoza', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Liczba zakażeń')
plt.title('Prognoza liczby zakażeń na następne 7 dni')
plt.legend()
plt.show()
