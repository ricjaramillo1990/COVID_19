# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import math
from pandas import Series
from keras.layers import BatchNormalization
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Nadam, Adamax, SGD
from keras.models import Sequential
from keras.layers import LSTM, Dense , Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from pandas import concat
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from math import sqrt
import sys
import ast
import os

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#SEED
np.random.seed(7)
tf.random.set_seed(123)

with open(sys.argv[1]) as f:
	pais_=f.read()

pais_=ast.literal_eval("{"+pais_[:-1]+"}")

with open(sys.argv[2]) as f:
	selectedFeatures=f.read()

selectedFeatures=ast.literal_eval("{"+selectedFeatures[:-1]+"}")

#CREATING FOLDER
fecha=datetime.today().strftime('%Y%m%d')

if len(os.path.basename(sys.argv[1])) > 30:
	pais=pais_
	if sys.argv[3] == 'deaths':
		path=("/home/org00004/LSTM/plots/"+fecha+"_rndmsearc_deaths_light")
		ext_variable='ConfirmedDeaths'
	elif sys.argv[3] == 'confirmed':
		path=("/home/org00004/LSTM/plots/"+fecha+"_rndmsearc_confirmed_light")
		ext_variable='ConfirmedCases'
else:
	path=("/home/org00004/LSTM/plots/"+fecha+"_rndmsearc_full")
	pais={}
	for z in list(selectedFeatures.keys()):
		if z in pais_:
			pais[z] = pais_[z]



Path(path).mkdir(parents=True, exist_ok=True)

#LOADING DATASET
dataset_global = pd.read_csv('https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv')
dataset_global["Date"]= pd.to_datetime(dataset_global["Date"], format='%Y%m%d')


#CREATING FILE
file1 = open(path+"/"+"log.txt","w")
file1.write("COUNTRY|TRAIN|TEST|DEATHS_FORECAST|COUNTS_DAY\n")

#Rules
c1={'N':3, 'F':1}
c2={'N':3, 'F':1}
c3={'N':2, 'F':1}
c4={'N':4, 'F':1}
c5={'N':2, 'F':1}
c6={'N':3, 'F':1}
c7={'N':2, 'F':1}
c8={'N':4, 'F':0}
e1={'N':2, 'F':1}
e2={'N':2, 'F':0}
h1={'N':2, 'F':1}
h2={'N':3, 'F':0}
h3={'N':2, 'F':0}

for key, value in pais.items():
	country=key
	time_step=pais[country][0]
	opt=pais[country][1]
	neurons=pais[country][2]
	lr=pais[country][3]
	num_epochs=pais[country][4]
	batch_size=pais[country][5]
	activation=pais[country][6]

	#print(country)
	file1.write(country+"|")

	dataset=dataset_global.loc[( (dataset_global.CountryName ==country) & (dataset_global.Date > "2020-03-01")),['CountryName','CountryCode','Date','C1_School closing','C1_Flag','C2_Workplace closing','C2_Flag','C3_Cancel public events','C3_Flag','C4_Restrictions on gatherings','C4_Flag','C5_Close public transport','C5_Flag','C6_Stay at home requirements','C6_Flag','C7_Restrictions on internal movement','C7_Flag','C8_International travel controls','E1_Income support','E1_Flag','E2_Debt/contract relief','E3_Fiscal measures','E4_International support','H1_Public information campaigns','H1_Flag','H2_Testing policy','H3_Contact tracing','H4_Emergency investment in healthcare','H5_Investment in vaccines','M1_Wildcard','ConfirmedCases','ConfirmedDeaths']].set_index('Date')
	dataset.columns = ['CountryName','CountryCode','C1','C1_Flag','C2','C2_Flag','C3','C3_Flag','C4','C4_Flag','C5','C5_Flag','C6','C6_Flag','C7','C7_Flag','C8','E1','E1_Flag','E2','E3','E4','H1','H1_Flag','H2','H3','H4','H5','M1','ConfirmedCases','ConfirmedDeaths']

	#Filling NaN due to uplead delay (Because data are updated on twice-weekly cycles, but not every country is updated in every cycle)
	dataset.fillna(method='ffill', inplace=True)
	#columns to look for missing values in deaths counts.
	dataset.dropna(subset=[ext_variable])

	dataset['C1_new']=round(100*((dataset['C1']-(0.5*(c1['F']-dataset['C1_Flag'])))/c1['N']),2)
	dataset.loc[(dataset['C1_new'] < 0), 'C1_new'] = 0
	dataset['C1_new']=dataset['C1_new'].fillna(0)

	dataset['C2_new']=round(100*((dataset['C2']-(0.5*(c2['F']-dataset['C2_Flag'])))/c2['N']),2)
	dataset.loc[(dataset['C2_new'] < 0), 'C2_new'] = 0
	dataset['C2_new']=dataset['C2_new'].fillna(0)

	dataset['C3_new']=round(100*((dataset['C3']-(0.5*(c3['F']-dataset['C3_Flag'])))/c3['N']),2)
	dataset.loc[(dataset['C3_new'] < 0), 'C3_new'] = 0
	dataset['C3_new']=dataset['C3_new'].fillna(0)

	dataset['C4_new']=round(100*((dataset['C4']-(0.5*(c4['F']-dataset['C4_Flag'])))/c4['N']),2)
	dataset.loc[(dataset['C4_new'] < 0), 'C4_new'] = 0
	dataset['C4_new']=dataset['C4_new'].fillna(0)

	dataset['C5_new']=round(100*((dataset['C5']-(0.5*(c5['F']-dataset['C5_Flag'])))/c5['N']),2)
	dataset.loc[(dataset['C5_new'] < 0), 'C5_new'] = 0
	dataset['C5_new']=dataset['C5_new'].fillna(0)

	dataset['C6_new']=round(100*((dataset['C6']-(0.5*(c6['F']-dataset['C6_Flag'])))/c6['N']),2)
	dataset.loc[(dataset['C6_new'] < 0), 'C6_new'] = 0
	dataset['C6_new']=dataset['C6_new'].fillna(0)

	dataset['C7_new']=round(100*((dataset['C7']-(0.5*(c7['F']-dataset['C7_Flag'])))/c7['N']),2)
	dataset.loc[(dataset['C7_new'] < 0), 'C7_new'] = 0
	dataset['C7_new']=dataset['C7_new'].fillna(0)

	dataset['C8_new']=round(100*((dataset['C8']-(0.5*(c8['F'])))/c8['N']),2)
	dataset.loc[(dataset['C8_new'] < 0), 'C8_new'] = 0
	dataset['C8_new']=dataset['C8_new'].fillna(0)

	dataset['E1_new']=round(100*((dataset['E1']-(0.5*(e1['F']-dataset['E1_Flag'])))/e1['N']),2)
	dataset.loc[(dataset['E1_new'] < 0), 'E1_new'] = 0
	dataset['E1_new']=dataset['E1_new'].fillna(0)

	dataset['E2_new']=round(100*((dataset['E2']-(0.5*(e2['F'])))/e2['N']),2)
	dataset.loc[(dataset['E2_new'] < 0), 'E2_new'] = 0
	dataset['E2_new']=dataset['E2_new'].fillna(0)

	dataset['H1_new']=round(100*((dataset['H1']-(0.5*(h1['F']-dataset['H1_Flag'])))/h1['N']),2)
	dataset.loc[(dataset['H1_new'] < 0), 'H1_new'] = 0
	dataset['H1_new']=dataset['H1_new'].fillna(0)

	dataset['H2_new']=round(100*((dataset['H2']-(0.5*(h2['F'])))/h2['N']),2)
	dataset.loc[(dataset['H2_new'] < 0), 'H2_new'] = 0
	dataset['H2_new']=dataset['H2_new'].fillna(0)

	dataset['H3_new']=round(100*((dataset['H3']-(0.5*(h3['F'])))/h3['N']),2)
	dataset.loc[(dataset['H3_new'] < 0), 'H3_new'] = 0
	dataset['H3_new']=dataset['H3_new'].fillna(0)

	#creating death toll per day
	dataset['Day_Count']=dataset[ext_variable].diff()
	dataset=dataset.drop(columns=[ext_variable])
	dataset.loc[(dataset['Day_Count'] < 0), 'Day_Count'] = 0

	if len(os.path.basename(sys.argv[2])) > 31:
		dataset['politic']=dataset[['C1_new','C2_new','C3_new','C4_new','C5_new','C6_new','C7_new','C8_new']].mean(axis = 1)
		dataset['economic']=dataset[['E1_new','E2_new']].mean(axis = 1)
		dataset['health']=dataset[['H1_new','H2_new','H3_new']].mean(axis = 1)
		dataset=dataset[['Day_Count','politic','economic','health']]

	else:
		dataset=dataset[['Day_Count','C5_new','E1_new','E2_new']]

	#final dataset
	param=list(selectedFeatures[country])
	#including death tolls
	param.insert(0,0)
	dataset=dataset.iloc[:,param]
	n_features = len(dataset.columns)
	#print(dataset.columns)
	#print(n_features)
	values = dataset.values

	# ensure all data is float
	values = values.astype('float32')
	# scaling
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	features= time_step*n_features
	# frame as supervised learning
	reframed = series_to_supervised(scaled, time_step, 1)

	# split into train and test sets
	values = reframed.values

	split_percent = 0.8
	split = int(split_percent*len(scaled))
	train = values[:split, :]
	test = values[split:, :]

	# split into input and outputs
	n_obs = time_step * n_features
	train_X, train_y = train[:, :n_obs], train[:, -n_features]
	test_X, test_y = test[:, :n_obs], test[:, -n_features]
	#print(train_X.shape, len(train_X), train_y.shape)
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], time_step, n_features))
	test_X = test_X.reshape((test_X.shape[0], time_step, n_features))
	#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

	#BUILDING THE MODEL
	model = Sequential()
	model.add(LSTM(neurons, activation=activation ,input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dropout(0.15))
	model.add(Dense(1))

	if opt ==0:
		opt=Adam
	elif opt ==1:
		opt=Nadam
	elif opt==2:
		opt=Adamax

	#COMPILING
	optimizer = opt(learning_rate=lr)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	#EARLY STOPPING
	es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=150)

	# fit network
	histo = model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=0, shuffle=False, callbacks=[es])

	plt.figure(figsize=(8,8))
	plt.plot(histo.history['loss'], label='train')
	plt.plot(histo.history['val_loss'], label='test')
	plt.legend()
	plt.title(country+' Loss   lr:'+str(lr)+", Steps:"+str(time_step)+", neurons:"+str(neurons)+", "+str(opt)[(str(opt).find("rs."))+3:-2]+", "+activation)
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.savefig(path+"/"+country+'_1loss')
	plt.clf()
	plt.close()
	#plt.show()


	# PREDICTION FOR TRAINING
	yhat_train = model.predict(train_X)
	train_X1 = train_X.reshape((train_X.shape[0], features))


	# invert scaling for actual
	train_y1 = train_y.reshape((len(train_y), 1))
	inv_y1 = concatenate((train_y1, train_X1[:, -(n_features-1):]), axis=1)
	inv_y1 = scaler.inverse_transform(inv_y1)
	inv_y1 = inv_y1[:,0]
	#print(inv_y1)

	# invert scaling for forecast
	inv_yhat1 = concatenate((yhat_train, train_X1[:, -(n_features-1):]), axis=1)
	inv_yhat1 = scaler.inverse_transform(inv_yhat1)
	inv_yhat1 = inv_yhat1[:,0]
	#print(inv_yhat1)

	# calculate RMSE
	trainScore = sqrt(mean_squared_error(inv_y1, inv_yhat1))
	#print('Train RMSE: %.3f' % trainScore)
	file1.write('%.2f|' % (trainScore))

	# PREDICTION FOR TEST
	yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], features))
	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
	# calculate RMSE
	testScore = sqrt(mean_squared_error(inv_y, inv_yhat))
	#print('Test RMSE: %.3f' % testScore)
	file1.write('%.2f|' % (testScore))

	#Creating plot for training
	train_date = dataset.index[1+time_step]
	train_dates = pd.date_range(train_date, periods=len(train_X)).strftime("%Y-%m-%d").tolist()

	trainingSet=pd.DataFrame(inv_yhat1, index=train_dates,columns=['ntrain']).reset_index()
	trainingSet['index']= trainingSet['index'].astype('datetime64[ns]').tolist()

	#Creating plot for test
	test_date = dataset.index[1+time_step+len(train_X)]
	test_dates = pd.date_range(test_date, periods=test_y.shape[0]).strftime("%Y-%m-%d").tolist()

	testSet=pd.DataFrame(inv_yhat, index=test_dates,columns=['ntest']).reset_index()
	testSet['index']= testSet['index'].astype('datetime64[ns]').tolist()

	x1 = trainingSet['index']
	y1 = trainingSet['ntrain']

	x2 = testSet['index']
	y2 = testSet['ntest']

	#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	plt.figure(figsize=(18,8))
	plt.plot(dataset['Day_Count'] , label='History')
	plt.plot(x1,y1 , label='Train')
	plt.plot(x2,y2 , label='Test')
	plt.title(country+' Model Train-Test')
	#plt.xticks(rotation=90, fontsize=7)
	#plt.grid(True)
	plt.legend(framealpha=1, frameon=True);
	plt.savefig(path+"/"+country+'_2training')
	plt.clf()
	plt.close()
	#plt.show()

	#FORECAST
	forecast=reframed.tail(1).values[0][-features:].reshape(1, time_step, n_features)
	#forecast

	#first forecast using reframed.tail(1)
	forecast_model=model.predict(forecast)

	# first forecast 
	prediction_list=[]
	forecast_X =forecast.reshape((1, features))

	#inverting scale
	inv_yhat_for = concatenate((forecast_model, forecast_X[:, -(n_features-1):]), axis=1)
	inv_yhat_for = scaler.inverse_transform(inv_yhat_for)
	inv_yhat_for = inv_yhat_for[:,0]
	#first value t+1
	prediction_list.append(inv_yhat_for[0])

	forecast=np.concatenate((forecast.reshape(-1),concatenate((forecast_model, forecast_X[:, -(n_features-1):]), axis=1).reshape(-1)))[-features:].reshape(1, time_step, n_features)

	num_prediction=15

	# -1  because we have already one prediction
	for x in range(0,num_prediction-1):
	  forecast=np.concatenate((forecast.reshape(-1),concatenate((forecast_model, forecast_X[:, -(n_features-1):]), axis=1).reshape(-1)))[-features:].reshape(1, time_step, n_features)
	  forecast_model=model.predict(forecast)
	  forecast_X =forecast.reshape((1, features))
	  inv_yhat_for = concatenate((forecast_model, forecast_X[:, -(n_features-1):]), axis=1)
	  inv_yhat_for = scaler.inverse_transform(inv_yhat_for)
	  inv_yhat_for = inv_yhat_for[:,0]
	  prediction_list.append(inv_yhat_for[0])

	last_date = dataset.index[-1]
	prediction_dates = pd.date_range(last_date, periods=num_prediction+1).strftime("%Y-%m-%d").tolist()
	
	#first entry of prediction list will be last date of the test set to add continuity to the plot
	prediction_list.insert(0, inv_y[-1])

	d_round=[round(c,2) for c in prediction_list]
	file1.write("%i|%s\n" % (sum(prediction_list),str(list(d_round))[1:-1]))

	#PLOTTING FORECAST
	fore=pd.DataFrame(prediction_list, index=prediction_dates,columns=['forecast']).reset_index()
	fore['index']= fore['index'].astype('datetime64[ns]').tolist()
	dayy=pd.DataFrame(dataset['Day_Count']).dropna().reset_index()
	
	x1 = dayy['Date']
	y1 = dayy['Day_Count']

	x2 = fore['index']
	y2 = fore['forecast']

	#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	plt.figure(figsize=(18,8))
	plt.plot(x1,y1 , label='History')
	plt.plot(x2,y2 , label='Forecast')
	plt.title(country+' Model Forecast')
	#plt.xticks(rotation=90, fontsize=7)
	#plt.grid(True)
	plt.legend(framealpha=1, frameon=True);
	plt.savefig(path+"/"+country+'_3forecast')
	plt.clf()
	plt.close()
	#plt.show()

file1.close()
