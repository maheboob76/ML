import sys
sys.executable

import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from random import random
from numpy import array
from sklearn.model_selection import train_test_split
from keras.metrics import binary_accuracy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

SAMPLES = 50000
CUM_PERIOD = 3 #sum of last 3 numbers
thres = 2


X1 = array([random() for _ in range(SAMPLES)])
X1_cum =  array( [ sum(X1[x-CUM_PERIOD:x]) for x in range(1, SAMPLES+1)  ] )
y = array([0 if x < thres else 1 for x in X1_cum])


df = pd.DataFrame({'X': X1, 'y':y})
# lagged data prep

cols = []
number_lags = 2
ldf = df
for lag in range(1, number_lags + 1):
    ldf['lag_' + str(lag)] = ldf.X.shift(lag)
    cols.append('lag_'+ str(number_lags +1 -lag) )

cols.extend(['X', 'y'])
#cols = ['lag_3', 'lag_2', 'lag_1', 'X', 'y']

#if you want numpy arrays with no null values: 
ldf = ldf.dropna()
#ldf[cols] = ldf
ldf = ldf.reindex(columns=cols)

X = ldf.values[:, :number_lags+1]
y = ldf.values[:, number_lags+1:]
X = X.reshape(X.shape[0],number_lags+1,1)
window_size = 3

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.2 ,shuffle = False, stratify = None)

# Define the LSTM model
model = Sequential()
model.add(LSTM( units=32, input_shape = (number_lags+1,1), return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(units=64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
model.summary()


start = time.time()
model.fit(train_X,train_y,batch_size=32,epochs=10)
print("> Compilation Time : ", time.time() - start)

# Doing a prediction on all the test data at once
preds = model.predict(test_X, batch_size=1, verbose=1)

#test_y = test_y.reshape(-1,1)
score, accu = model.evaluate(test_X, test_y, verbose=1)
print('Test score:', score)
print('Test accuracy:', accu)
#actuals = test_y

r = np.round(preds).astype(int)
#print('keras function accuracy: ' , binary_accuracy(test_y.astype(float), preds) )
print('sklearn function accuracy: ' , accuracy_score(test_y, r,  normalize=True) )


print( 'Accuracy:', accuracy_score(test_y, r) )
print('F1 score:', f1_score(test_y, r, average='weighted'))
print( 'Recall:', recall_score(test_y, r,   average='weighted') )
print( 'Precision:', precision_score(test_y, r,     average='weighted') )
print( '\n clasification report:\n', classification_report(test_y, r) )
print( '\n confussion matrix:\n',confusion_matrix(test_y, r) )


preds = preds.reshape(preds.shape[0])
#test_X = test_X.reshape(test_X.shape[0],1)
test_y = test_y.reshape(test_y.shape[0])

result = pd.DataFrame({'actuals': np.round(test_y), 'pred': preds})

#brand new prediction
#thres = 2 # change this to force errors in data
SAMPLES = 20000
X1 = array([random() for _ in range(SAMPLES)])
X1_cum =  array( [ sum(X1[x-CUM_PERIOD:x]) for x in range(1, SAMPLES+1)  ] )
y = array([0 if x < thres else 1 for x in X1_cum])


df = pd.DataFrame({'X': X1, 'y':y})
# lagged data prep

cols = []
number_lags = 2
ldf = df
for lag in range(1, number_lags + 1):
    ldf['lag_' + str(lag)] = ldf.X.shift(lag)
    cols.append('lag_'+ str(number_lags +1 -lag) )

cols.extend(['X', 'y'])
#cols = ['lag_3', 'lag_2', 'lag_1', 'X', 'y']

#if you want numpy arrays with no null values: 
ldf = ldf.dropna()
#ldf[cols] = ldf
ldf = ldf.reindex(columns=cols)

X = ldf.values[:, :number_lags+1]
y = ldf.values[:, number_lags+1:]
X = X.reshape(X.shape[0],number_lags+1,1)

preds = model.predict(X, batch_size=1, verbose=1)
r = np.round(preds).astype(int)


print( 'Accuracy:', accuracy_score(y, r) )
print('F1 score:', f1_score(y, r, average='weighted'))
print( 'Recall:', recall_score(y, r,   average='weighted') )
print( 'Precision:', precision_score(y, r,     average='weighted') )
print( '\n clasification report:\n', classification_report(y, r) )
print( '\n confussion matrix:\n',confusion_matrix(y, r) )

