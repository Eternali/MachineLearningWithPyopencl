import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

data = np.array([[i for i in range(100)]], dtype=np.float32).reshape((1, 1, 100))
target = np.array([[i for i in range(50, 150)]], dtype=np.float32).reshape((1, 1, 100))

test = np.array([[i for i in range(100)]], dtype=np.float32).reshape((1, 1, 100))
test_target = np.array([[i for i in range(50, 150)]], dtype=np.float32).reshape((1, 1, 100)) 

model = Sequential()
model.add(LSTM(100, input_shape=(1, 100), return_sequences=True))
model.add(Dense(100))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.fit(data, target, nb_epoch=10000, batch_size=1, verbose=2, validation_data=(test, test_target))

# display a prediction
print(model.predict(test))
