import pandas as pd
from tensorflow.keras.layers import Dense
from keras.layers import Input
import tensorflow as tf
from tensorflow import keras
from keras import Model
from tensorflow.keras.models import Sequential
from keras.layers import concatenate
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

def build_branch(n_nodes, n_hidden, activation, input):
    branch = Dense(n_nodes, activation=activation)(input)
    for i in range(n_hidden - 1):
        branch = Dense(n_nodes, activation=activation)(branch)
    branch = Dense(1, activation='linear')(branch)
    return branch

data = pd.read_csv('data.csv')
X = data.iloc[:, [0,1,2]] # input data
y = data.iloc[:, [3]] # output data

# this is the input split that was found to produce the lowest error in the previous script
input_split = [[0,1], [2]]

# build branch model
input_index_branch_1 = input_split[0]
input_index_branch_2 = input_split[1]
input_1 = Input(shape=[len(input_index_branch_1)])
input_2 = Input(shape=[len(input_index_branch_2)])
branch_1 = build_branch(100, 3, 'tanh', input_1)
combined_z_input2 = tf.keras.layers.Concatenate()([branch_1, input_2])
branch_2 = build_branch(100, n_hidden=3, activation='tanh', input=combined_z_input2)
full_model = Model(inputs=[input_1, input_2], outputs=branch_2)

# compile and train branch model
full_model.compile(optimizer='adam', loss='mse')
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
print("Training model...")
full_model.fit([X.iloc[:, input_index_branch_1], X.iloc[:, input_index_branch_2]], y, epochs=200, batch_size=256, callbacks=[callback], validation_split=0.2, verbose=0)

# now we want to interpret the model by visualizing the mapping of each branch

# first, create a replicate model of branch 1 by importing the branch 1 layers
model_branch_1 = Sequential()
model_branch_1.add(full_model.get_layer('dense'))
model_branch_1.add(full_model.get_layer('dense_1'))
model_branch_1.add(full_model.get_layer('dense_2'))
model_branch_1.add(full_model.get_layer('dense_3'))
model_branch_1.compile(optimizer='adam', loss='mse')

# Predict the output of just branch 1
z = model_branch_1.predict(X.iloc[:, input_index_branch_1])

# Create a replicate of branch 2

model_branch_2 = Sequential()
model_branch_2.add(full_model.get_layer('dense_4'))
model_branch_2.add(full_model.get_layer('dense_5'))
model_branch_2.add(full_model.get_layer('dense_6'))
model_branch_2.add(full_model.get_layer('dense_7'))
model_branch_2.compile(optimizer='adam', loss='mse')

# Predict the final output, which is the output of branch 2
# Note that branch 2 needs the output from branch 1 as one of its inputs
input_to_branch2 = concatenate([z, X.iloc[:, input_index_branch_2]])
y_pred = model_branch_2.predict(input_to_branch2)

# plot branch 1

plt.figure()
plt.scatter(X.iloc[:,1], z, c=X.iloc[:,0], cmap='viridis')
plt.xlabel('x')
plt.ylabel('z')
plt.colorbar(label='A')
plt.title('Branch 1')
plt.tight_layout()

# plot branch 2

plt.figure()
plt.scatter(X.iloc[:,2], y_pred, c=z, cmap='viridis')
plt.xlabel('B')
plt.ylabel('predicted y')
plt.colorbar(label='z')
plt.title('Branch 2')
plt.tight_layout()

plt.show()


