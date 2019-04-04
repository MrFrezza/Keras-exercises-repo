# Define um rede neural para separa dados de dois grupos de informações

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Helper functions

# Para plotar
def plot_data(pl, x, y):
    # plot a classe onde y==0
    pl.plot(x[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    # plot a classe onde y==1
    pl.plot(x[y==1, 0], x[y==1, 1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl

# Função para desenhar a linha limite de decisão que for definida
def plot_decision_boundary(model, x, y):

    amin, bmin = x.min(axis=0) - 0.1
    amax, bmax = x.max(axis=0) + 0.1   
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # fazer as predições com o modelo e remodelar a saída para que possa ser plotado
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12,8))
    #plotar o contour
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)
    # plotar os conjuntos de dados
    plot_data(plt, x, y)

    return plt


# Generate some data blobs. Data will be either 0 or 1 when 2 is the number os centers.
# X is a [number of samples, 2] sized array. X[sample] contains its x,y position of the sample in the space
# ex: X[1] = [1.342, -2.3], X[2] = [-4.342, 2.12]
# y is a [number of sample] sized array. y[sample] contains the class index (ie. 0 or 1 when there are 2 centers)
# ex: y[1] = 0, y[´1] = 1 
X, y = make_blobs(n_samples=1000, centers=2, random_state=42) 

pl = plot_data(plt, X, y)
pl.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creates Keras Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Simple Sequential model
model = Sequential()
# Add a dense fully connected layer with 1 neuron. Using input_shape=(2,) says the input will
#   be arrays of the form (*,2). The first dimension will be an unspecified
#    nummber of batches (rows) of data. The second dimension is 2 which are the X, Y positions of each data element.
#    The sigmoid activation function is used to return 0 or 1, signifying the data 
#    cluster the position is predicted to belong to.     
model.add(Dense(1, input_shape=(2,), activation="sigmoid"))
# Compile the model. Minimize crossentropy for a binary. Maximize for accuracy
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
# Fit the model with the data from make_blobs. Make 100 cycles through the data
#   Set verbose to 0 to supress progress messages
model.fit(X_train, y_train, epochs=100, verbose=0)

eval_result = model.evaluate(X_test, y_test)

# Print test accuracy
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])
# Plot the dicision boundary
plot_decision_boundary(model, X, y).show()