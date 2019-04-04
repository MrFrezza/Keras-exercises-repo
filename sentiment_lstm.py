# Classificar reviews do IMDB baseado no sentimento, positivo ou negativo.
# IMDB database do keras

# As reviews já foram pre processadas. Portanto a segunda palavra mais utilizada será substituida
# por '2', a terceira por '3' e assim por diante

from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb

# Supress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_WORDS = 6000        # As 'N' palavras mais frequentes
SKIP_TOP = 0            # Pula as palavras mais comuns do topo
MAX_REVIEW_LEN = 400    # MAX quantidade de palavras em uma review

# carregar as informações pre-processadas
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS, skip_top=SKIP_TOP)

# Print sample
#print("enconded word sequence: ", x_train[3])

# Garantir que possuam o mesmo tamanho
x_train = sequence.pad_sequences(x_train, maxlen= MAX_REVIEW_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_REVIEW_LEN)

print('x_train.shape: ', x_train.shape, 'x_test.shape: ', x_test.shape)

model = Sequential()
# Cria um vetor denso para permitir que identifique itens iguais
model.add(Embedding(NUM_WORDS, 64))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# Constantes para treinamento
BATCH_SIZE = 24
EPOCHS = 5

cbk_early_stopping = EarlyStopping(monitor='val_acc', mode='max')

model.fit(x_train, y_train, BATCH_SIZE, epochs=EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=[cbk_early_stopping] )
