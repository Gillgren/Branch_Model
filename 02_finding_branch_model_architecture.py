import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense
from keras.layers import Input
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras import Model
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# A function that builds a branch, similar to how a neural network is built
def build_branch(n_nodes, n_hidden, activation, input):
    branch = Dense(n_nodes, activation=activation)(input)
    for i in range(n_hidden - 1):
        branch = Dense(n_nodes, activation=activation)(branch)
    branch = Dense(1, activation='linear')(branch)
    return branch

data = pd.read_csv('data.csv')
X = data.iloc[:, [0,1,2]] # input data
y = data.iloc[:, [3]] # output data

# Setup KFold object to enable cross validation evaluation of the error later in the code

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(X)

# Setup the three ways that the three inputs can be split into the different branches
# This step can be automated for more complicated problems

input_splits = []
input_splits.append([[0,1], [2]])
input_splits.append([[0,2], [1]])
input_splits.append([[1,2], [0]])

print("Training models...")
mse_input_splits = []
for i in range(len(input_splits)):
    mse_vector = [] # the mse vector for each way to split the inputs into branches
    for train_index, test_index in kf.split(X):
        # split data into temporary training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

        # build branch model
        input_index_branch_1 = input_splits[i][0]
        input_index_branch_2 = input_splits[i][1]
        input_1 = Input(shape=[len(input_index_branch_1)])
        input_2 = Input(shape=[len(input_index_branch_2)])
        branch_1 = build_branch(100, 3, 'tanh', input_1)
        combined_z_input2 = tf.keras.layers.Concatenate()([branch_1, input_2])
        branch_2 = build_branch(100, n_hidden=3, activation='tanh', input=combined_z_input2)
        full_model = Model(inputs=[input_1, input_2], outputs=branch_2)

        # compile and train branch model
        full_model.compile(optimizer='adam', loss='mse')
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        full_model.fit([X_train.iloc[:, input_index_branch_1], X_train.iloc[:, input_index_branch_2]], Y_train, epochs=200, batch_size=256, callbacks=[callback], validation_split=0.2, verbose=0)

        # Predict on test set
        Y_test_pred = full_model.predict([X_test.iloc[:, input_index_branch_1], X_test.iloc[:, input_index_branch_2]])
        mse = mean_squared_error(Y_test, Y_test_pred)
        mse_vector.append(mse)

    # store mean mse for each way to split the inputs into the branches
    mse_input_splits.append(np.mean(mse_vector))

# Excecuting the code results in the following mse for the different input splits:

print("The MSE for the different input splits into the different branches")
print("[[0,1], [2]]: " + str(mse_input_splits[0]))
print("[[0,2], [1]]: " + str(mse_input_splits[1]))
print("[[1,2], [0]]: " + str(mse_input_splits[2]))

plt.scatter([1,2,3],mse_input_splits, label='Branch Models', c='black')
plt.plot([0.8,4], [7.443053774565674e-05, 7.443053774565674e-05], label='regular NN') # benchmark error of regular dense NN from script 01
plt.yscale("log")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend()
plt.xticks([])
plt.xlabel('Input splits')
plt.text(1.03,mse_input_splits[0], "(A, x) (B)")
plt.text(2.03,mse_input_splits[1], "(A, B) (x)")
plt.text(3.03,mse_input_splits[2], "(x, B) (A)")
plt.xlim(0.8,4)
plt.tight_layout()
plt.show()

# By running this script, we obtain the following MSE for the different splits:

#[[0,1], [2]]: 0.0003159065520950948 (clear winner)
#[[0,2], [1]]: 0.03465058993819779
#[[1,2], [0]]: 0.013620949511121683

# as stated before, these values may vary slightly each time the code is executed
