import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow import keras
import numpy as np

def build_NN(n_nodes, n_layers, activation):
    model = Sequential()
    model.add(Dense(n_nodes, input_dim=3, activation=activation))
    for i in range(n_layers-1):
        model.add(Dense(n_nodes, activation=activation))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# true underlying function: y = Asin(10x) + B
# data indices:
# 0: A
# 1: x
# 2: B
# 3: y

data = pd.read_csv('data.csv')
X = data.iloc[:, [0,1,2]] # input data
y = data.iloc[:, [3]] # output data

# Setup KFold object to enable cross validation evaluation of the error later in the code

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(X)

# Loop that trains a neural network for each data split (we split the data to enable cross validation)

mse_vector = [] # mse = mean squared error

print("Training models...")
for train_index, test_index in kf.split(X):
    # split data into temporary training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

    # build and train model
    model = build_NN(100, 3, 'tanh')
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=200, batch_size=256, callbacks=[callback], validation_split=0.2, verbose=0)

    # Predict on temporary test set and store mse value
    Y_test_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_test_pred)
    mse_vector.append(mse)

# print both the entire vector (to make sure that no run stands out with a significantly higher error) and the mean of the mse
print("Test set mse vector for all data splits: " + str(mse_vector))
print("Mean test set mse: " + str(np.mean(mse_vector)))

# By running this script, we achieve a mean of the mean squared error of:
# 7.443053774565674e-05
# This value may vary slightly each time the code is executed, as the neural network weights are randomly initialized
# Nevertheless, this value is saved to be used as the benchmark error when using the Branch Model