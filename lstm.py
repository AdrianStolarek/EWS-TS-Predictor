import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('DATA_ALL.csv')
df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
df = df.sort_values('Data')
data = df['liczba_wszystkich_zakazen'].values

df['day_of_week'] = df['Data'].dt.dayofweek
data_with_features = df[['liczba_wszystkich_zakazen', 'day_of_week']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_with_features)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
#
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(look_back, X.shape[2]))))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#
history = model.fit(X_train, Y_train, epochs=100, batch_size=7, verbose=2, validation_data=(X_val, Y_val), callbacks=[early_stopping])

def predict_future(data, model, look_back, days):
    predictions = []
    current_batch = data[-look_back:]
    current_batch = current_batch.reshape((1, look_back, data.shape[1]))

    for _ in range(days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred[0])
        current_pred_features = np.zeros((1, data.shape[1]))
        current_pred_features[0, 0] = current_pred  # Only the first feature is the prediction
        current_batch = np.append(current_batch[:, 1:, :], [current_pred_features], axis=1)

    return predictions

future_days = 7
future_predictions = predict_future(scaled_data, model, look_back, future_days)
future_predictions = np.array(future_predictions).reshape(-1, 1)  # Reshape to 2D array
future_predictions = scaler.inverse_transform(np.concatenate([future_predictions, np.zeros((future_days, 1))], axis=1))[:, 0]
future_predictions = np.round(future_predictions).astype(int)  # Ensure predictions are whole numbers

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
