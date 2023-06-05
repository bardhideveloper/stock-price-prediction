
#IMPORTIMI I LIBRARIVE TE NEVOJSHME PER STOCK PRICE PREDICTION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#LEXIMI I DATASETIT TESLA , PRINTIMI I DISA INFORMATAVE KRYESORE TE DATASETIT
df = pd.read_csv('TSLA.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
##
##close_prices = df['Close']
##values = close_prices.values
##training_data_len = math.ceil(len(values)* 0.8)
##
##scaler = MinMaxScaler(feature_range=(0,1))
##scaled_data = scaler.fit_transform(values.reshape(-1,1))
##train_data = scaled_data[0: training_data_len, :]
##
##x_train = []
##y_train = []
##
##for i in range(60, len(train_data)):
##    x_train.append(train_data[i-60:i, 0])
##    y_train.append(train_data[i, 0])
##    
##x_train, y_train = np.array(x_train), np.array(y_train)
##x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
##
##test_data = scaled_data[training_data_len-60: , : ]
##x_test = []
##y_test = values[training_data_len:]
##
##for i in range(60, len(test_data)):
##  x_test.append(test_data[i-60:i, 0])
##
##x_test = np.array(x_test)
##x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
##
##model = keras.Sequential()
##model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
##model.add(layers.LSTM(100, return_sequences=False))
##model.add(layers.Dense(25))
##model.add(layers.Dense(1))
##model.summary()
##
####model.compile(optimizer='adam', loss='mean_squared_error')
####model.fit(x_train, y_train, batch_size= 1, epochs=3)
##
##predictions = model.predict(x_test)
##predictions = scaler.inverse_transform(predictions)
##rmse = np.sqrt(np.mean(predictions - y_test)**2)
##rmse
##
##data = df.filter(['Close'])
##train = data[:training_data_len]
##validation = data[training_data_len:]
##validation['Predictions'] = predictions
##plt.figure(figsize=(16,8))
##plt.title('Model')
##plt.xlabel('Date')
##plt.ylabel('Close Price USD ($)')
##plt.plot(train)
##plt.plot(validation[['Close', 'Predictions']])
##plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
##plt.show()


#CMIMET E TESLAS , TREGOJNE NJE PRIRJE RRITESE , ME POSHTE PARAQITET GRAFIKU I CMIMIT TE MBYLLJES SE AKSIONEVE
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()


#KOLONAT 'Close' DHE 'Adj Close' JANE TE NJEJTA , ATEHERE KOLONEN 'Adj Close' E FSHIJME NGA DATASETI
df.head()
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)


#KONTROLLIMI NESE KA TE PRANISHME VLERA NULL NE DATASET
df.isnull().sum()


#VIZATIMI I GRAFIKUT TE SHPERNDARJES PER VECORITE E VAZHDUESHME NE DATASET 
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10)) 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()


#VIZATOJME SKEMAT E KUTISE , VETEM TE DHENAT E VELLIMIT PERMBAJNE VLERA TE JASHTME,
#TE DHENAT NE PJESEN TJETER TE KOLONAVE JANE TE LIRA NGA QDO PERCAKTIM I JASHTEM
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()


#INXHINIERIA E VECORIVE - NDIHMON NE NXJERREN E VECORIVE TE VLEFSHME NGA ATO EKZISTUESE 
#NXJERRIM NGA KOLONA 'Data' tre kolona te reja : 'year' , 'month' , 'day'
df['Date']= pd.to_datetime(df['Date'],format='%m/%d/%Y')
df['year']= df['Date'].dt.year
df['month']= df['Date'].dt.month
df['day']= df['Date'].dt.day 
print(df.head())


#1/4 PERCAKTOHET SI NJE GRUP PREJ TRE MUAJSH , KOMPANITE PERGADITIN REZULTATET TREMUJORE
#NDIKOJNE SHUME NE CMIMET E AKSIONEVE PRANDAJ E KEMI SHTUAR KETE VECORI.
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
print(df.head())


#PARAQITJA E NJE GRAFIKU ME SHTYLLA , KU CMIMET E AKSIONEVE JANE DYFISHUAR NGA VITI 2013 DERI NE VITIN 2014
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()


#VEZHGIME TE RENDESISHME TE TE DHENAVE
#CMIMET JANE ME TE LARTA NE MUAJT QE JANE NE FUND TE TREMUJORIT NE KRAHASIM ME ATO TE MUAJVE TE JO TE FUNDIT TE TREMUJORIT
#VELLIMI I TREGTIMEVE ESHTE ME I ULET NE MUAJT QE JANE NE FUND TE TREMUJORIT
print(df.groupby('is_quarter_end').mean())


#KEMI SHTUAR DISA KOLONA TE REJA - NDIHMESE PER TRAJNIM TE MODELIT , KONTROLLIMI PERMES GRAFIKUT NESE OBJEKTIVI ESHTE I BALANCUAR
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()


#KUR SHTOJME VECORI , SIGUROHEMI QE NUK JANE TE NDERLIDHURA , PASI QE NUK NA NDIHMOJNE NE PROCESIN E ALGORITMIT
#VECORITE E SHTUARA NUK JANE SHUME TE LIDHURA ME NJERA TJETREN , JEMI MIRE TE NDERTOJME MODELIN TONE
plt.figure(figsize=(10, 10)) 
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()


#NDARJA DHE NORMALZIMI I TE DHENAVE
#NORMALIZIMI I TE DHENAVE NA QON NE TRAJNIM TE QENDRUESHEM DHE TE SHPEJTE
#TE DHENAT JANE NDARE NE DY PJESE ME NJE RAPORT 90/10 , VLERSOJME PERFORMANCEN E MODELIT NE TE DHENA TE PADUKSHME
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target'] 
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_test.shape)


#ZHVILLIMI DHE VLERESIMI I MODELIT
#REGRESIONI LOGJIK 
LR = LogisticRegression()
LR.fit(X_train, Y_train)

print(f'{LR} : ')
print('Training Accuracy : ', metrics.roc_auc_score(Y_train, LR.predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(Y_test, LR.predict_proba(X_test)[:,1]))
print()

#MATRICA E KONFUZIONIT PER TE DHENAT E VERTETIMIT LR
metrics.plot_confusion_matrix(LR, X_test, Y_test)
plt.show()

#SUPPORT VECTOR MACHINE 
SVC = SVC(kernel='poly', probability=True)
SVC.fit(X_train, Y_train)

print(f'{SVC} : ')
print('Training Accuracy : ', metrics.roc_auc_score(Y_train, SVC.predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(Y_test, SVC.predict_proba(X_test)[:,1]))
print()

#MATRICA E KONFUZIONIT PER TE DHENAT E VERTETIMIT SVC
metrics.plot_confusion_matrix(SVC, X_test, Y_test)
plt.show()

#XGBCLASSIFIER
XGB = XGBClassifier()
XGB.fit(X_train , Y_train)

print(f'{XGB} : ')
print('Training Accuracy : ', metrics.roc_auc_score(Y_train, XGB.predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(Y_test, XGB.predict_proba(X_test)[:,1]))
print()

#MATRICA E KONFUZIONIT PER TE DHENAT E VERTETIMIT LR
metrics.plot_confusion_matrix(XGB, X_test, Y_test)
plt.show()


