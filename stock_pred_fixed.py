import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# BUG 1 FIXED: %matplotlib inline is a Jupyter magic command.
# It causes a SyntaxError when running as a plain .py script.
# Replaced with matplotlib.use('Agg') which works in both
# script and notebook environments (saves figures to file instead
# of trying to open a GUI window).
import matplotlib
matplotlib.use('Agg')

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.preprocessing import MinMaxScaler

# BUG 2 FIXED: Sequential imports from keras are missing Dropout.
# It was listed in the import but never used — not a crash bug,
# but cleaned up to match what the model actually uses.
from keras.models import Sequential
from keras.layers import LSTM, Dense

scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
df.head()

df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']

plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')

data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

# BUG 3 FIXED: Chained indexing new_dataset["Date"][i] = ... is unsafe.
# In newer pandas this raises SettingWithCopyWarning and can silently
# do nothing. Use .loc[] for reliable assignment.
for i in range(0, len(data)):
    new_dataset.loc[i, "Date"] = data['Date'].iloc[i]
    new_dataset.loc[i, "Close"] = data["Close"].iloc[i]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

final_dataset = new_dataset.values

train_data = final_dataset[0:987, :]
valid_data = final_dataset[987:, :]

scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# BUG 4 FIXED: model.predict() was called but the variable is named lstm_model.
# `model` is undefined at this point — this is a NameError crash.
# Changed to lstm_model.predict() to match the variable defined above.
closing_price = lstm_model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

lstm_model.save("saved_lstm_model.h5")

train_data = new_dataset[:987]
valid_data = new_dataset[987:]

# BUG 5 FIXED: valid_data['Predictions'] = prediction_closing
# `prediction_closing` is undefined — the correct variable is `closing_price`.
# Also, closing_price is shape (n,1); flatten to 1D to avoid DataFrame
# assignment shape mismatch.
valid_data['Predictions'] = closing_price.flatten()

plt.figure(figsize=(16, 8))
plt.plot(train_data["Close"], label="Train")
plt.plot(valid_data[['Close', "Predictions"]])
plt.savefig("prediction_plot.png")  # Save instead of show (works outside Jupyter)
plt.close()
