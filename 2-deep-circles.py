# Define um rede neural para separa dados de dois grupos de informações

from sklearn.datasets import make_circles
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

X, y = make_circles(n_samples=1000, factor=0.6, noise=0.1 ,random_state=42) 

pl = plot_data(plt, X, y)
pl.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creates Keras Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping

# Simple Sequential model
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation="tanh", name="Hidden-1"))
model.add(Dense(4, activation="tanh", name="Hidden-2"))

model.add(Dense(1, activation="sigmoid", name="Output_layer"))
model.summary()

#Plota uma imagem com a estrutura da rede neural passada
plot_model(model, to_file="model.png", show_layer_names=True, show_shapes=True)

my_callbacks = [EarlyStopping(monitor='val_acc', patience=5, mode=max)]

# Compile the model. Minimize crossentropy for a binary. Maximize for accuracy
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
# Fit the model with the data from make_blobs. Make 100 cycles through the data
#   Set verbose to 0 to supress progress messages
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))

eval_result = model.evaluate(X_test, y_test)

# Print test accuracy
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])
# Plot the dicision boundary
plot_decision_boundary(model, X, y).show()