# -*- coding: utf-8 -*-
"""Feature Selection

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZGN0pu4Ktd8UX3rd0fhR-HfIvLdU0dlW
"""

import numpy
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
from keras.optimizers import Adam, Nadam, Adamax
import pandas as pd
from pandas import DataFrame
from pandas import concat
import tensorflow as tf
from pandas import concat
from numpy import concatenate
from keras.wrappers.scikit_learn import KerasRegressor
from math import sqrt
import random
from datetime import datetime
from itertools import combinations
import math
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

numpy.random.seed(7)
tf.random.set_seed(123)

with open(sys.argv[1]) as f:
	pais=f.read()

pais=ast.literal_eval("{"+pais[:-1]+"}")


#CREATING FOLDER
fecha=datetime.today().strftime('%Y%m%d')
path=("/home/org00004/LSTM/")

if sys.argv[2] == 'deaths':
	ext_variable='ConfirmedDeaths'
elif sys.argv[2] == 'confirmed':
	ext_variable='ConfirmedCases'

#CREATING FILE
if len(os.path.basename(sys.argv[1])) > 28:
	if ext_variable == 'ConfirmedDeaths':
		file1 = open(path+fecha+"_random_deaths_light.txt","w")
	elif ext_variable == 'ConfirmedCases':
		file1 = open(path+fecha+"_random_confir_light.txt","w")
else:
	file1 = open(path+fecha+"_random_search_full.txt","w")
	pais= dict(random.sample(list(pais.items()), k=2))

#DATASET
dataset_global = pd.read_csv('https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv')
dataset_global["Date"]= pd.to_datetime(dataset_global["Date"], format='%Y%m%d')

#RULES
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

	if opt ==0:
	  opt=Adam
	elif opt ==1:
	  opt=Nadam
	elif opt==2:
	  opt=Adamax
	print(country)
	dataset=dataset_global.loc[( (dataset_global.CountryName ==country) & (dataset_global.Date > "2020-01-01")),['CountryName','CountryCode','Date','C1_School closing','C1_Flag','C2_Workplace closing','C2_Flag','C3_Cancel public events','C3_Flag','C4_Restrictions on gatherings','C4_Flag','C5_Close public transport','C5_Flag','C6_Stay at home requirements','C6_Flag','C7_Restrictions on internal movement','C7_Flag','C8_International travel controls','E1_Income support','E1_Flag','E2_Debt/contract relief','E3_Fiscal measures','E4_International support','H1_Public information campaigns','H1_Flag','H2_Testing policy','H3_Contact tracing','H4_Emergency investment in healthcare','H5_Investment in vaccines','M1_Wildcard','ConfirmedCases','ConfirmedDeaths']].set_index('Date')
	dataset.columns = ['CountryName','CountryCode','C1','C1_Flag','C2','C2_Flag','C3','C3_Flag','C4','C4_Flag','C5','C5_Flag','C6','C6_Flag','C7','C7_Flag','C8','E1','E1_Flag','E2','E3','E4','H1','H1_Flag','H2','H3','H4','H5','M1','ConfirmedCases','ConfirmedDeaths']

	#filling NaN due to uplead delay (Because data are updated on twice-weekly cycles, but not every country is updated in every cycle)
	dataset.fillna(method='ffill', inplace=True)

	#columns to look for missing values.
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

	if len(os.path.basename(sys.argv[1])) > 28:
		dataset['politic']=dataset[['C1_new','C2_new','C3_new','C4_new','C5_new','C6_new','C7_new','C8_new']].mean(axis = 1)
		dataset['economic']=dataset[['E1_new','E2_new']].mean(axis = 1)
		dataset['health']=dataset[['H1_new','H2_new','H3_new']].mean(axis = 1)
		dataset=dataset[['Day_Count','politic','economic','health']]
		factor=0.5
	else:
		dataset=dataset[['Day_Count','C5_new','E1_new','E2_new']]
		factor=0.05
	dataset=dataset.dropna()

	values = dataset.values

	# ensure all data is float
	values = values.astype('float32')
	# scaling
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	# specify the number of time_steps
	n_features = len(dataset.columns)
	features= time_step*n_features
	# frame as supervised learning
	reframed = series_to_supervised(scaled, time_step, 1)
	#print(reframed.shape)

	reframed.tail()

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
	file1.write("\""+country+"\":")
	#print(country+"\n")
	##HILL CLIMBING
	best_loop=0
	winner=0
	for valor in range(1,n_features):
		best=0
		lista=[]
		tomados=valor
		# COMPUTING ALL POSSIBLE COMBINATIONS. RANGE STARTS IN 1 DUE TO 1 COLUMN IS THE DEATH TOLLS
		combinaciones=[i for i in list(combinations(list(range(1,n_features)), tomados))]
		#print("%ic%i" % (n_features-1,tomados))
		# JUST TESTING THE 50% OF RANDOM VARIANTS PER COMBINATION OPTION
		taken=math.ceil(len(combinaciones)*factor)
		chosen=random.sample(combinaciones, k=taken)
		#print("taking: %i" % (taken))
		#print("indexes selected: %s" % (chosen))
		for s in range(0,len(chosen)):
			def create_model(neurons=neurons,activation=activation, learn_rate=lr , opt=opt):
				model = Sequential()
				model.add(LSTM(units=neurons, activation=activation ,input_shape=(train_X.shape[1], tomados+1)))
				model.add(Dropout(0.15))
				model.add(Dense(units=1))
				# COMPILING
				lr=learn_rate
				optimizer = opt(learning_rate=lr)
				model.compile(loss='mean_squared_error', optimizer=optimizer)
				return model
			kears_estimator = KerasRegressor(build_fn=create_model,epochs=num_epochs, batch_size=batch_size, verbose=0)
			#RESHAPING TRAINING SET
			#train_new=numpy.concatenate((train_X[:,:,0],train_X[:,:,wey])).reshape((train_X.shape[0], time_step, tomados+1))#cambio aqui
			train_new=train_X[:,:,0].reshape((train_X.shape[0], time_step, 1))
			for wey in chosen[s]:
			  #APPENDING COLUMNS ACCORDING TO THE CHOSEN FEATURES LENGTH FOR TRAIN SET
			  train_new=numpy.concatenate((train_new,train_X[:,:,wey].reshape((train_X.shape[0], time_step, 1))),axis=2)
			#RESHAPING TEST SET
			#test_new=numpy.concatenate((test_X[:,:,0],test_X[:,:,wey])).reshape((test_X.shape[0], time_step, tomados+1))#cambio
			test_new=test_X[:,:,0].reshape((test_X.shape[0], time_step, 1))
			for wey in chosen[s]:
			  #APPENDING COLUMNS ACCORDING TO THE CHOSEN FEATURES LENGTH FOR TEST SET
			  test_new=numpy.concatenate((test_new,test_X[:,:,wey].reshape((test_X.shape[0], time_step, 1))),axis=2)

			# FITTING DATASET
			history = kears_estimator.fit(train_new, train_y, epochs=num_epochs, batch_size=batch_size, verbose=0, shuffle=False)
			score=kears_estimator.score(test_new, test_y)

			# # MEASURING TRAIN
			# yhat_train = kears_estimator.predict(train_new)
			# train_X1 = train_X.reshape((train_X.shape[0], features))
			# # invert scaling for actual / appending same colums from train to not have problem with inverse scale
			# train_y1 = train_y.reshape((len(train_y), 1))
			# inv_y1 = concatenate((train_y1, train_X1[:, -(n_features-1):]), axis=1)
			# inv_y1 = scaler.inverse_transform(inv_y1)
			# inv_y1 = inv_y1[:,0]
			# # invert scaling for forecast/ appending same colums from train to not have problem with inverse scale
			# inv_yhat1 = concatenate((yhat_train.reshape(train_X.shape[0],1), train_X1[:, -(n_features-1):]), axis=1)
			# inv_yhat1 = scaler.inverse_transform(inv_yhat1)
			# inv_yhat1 = inv_yhat1[:,0]
			# # calculate RMSE
			# rmse_t = sqrt(mean_squared_error(inv_y1, inv_yhat1))
			# #print('Train RMSE: %.3f' % rmse_t)

			# MEASURING TEST
			yhat = kears_estimator.predict(test_new)
			test_X_ = test_X.reshape((test_X.shape[0], features))
			# invert scaling for actual/ appending same colums from test to not have problem with inverse scale
			test_y = test_y.reshape((len(test_y), 1))
			inv_y = concatenate((test_y, test_X_[:, -(n_features-1):]), axis=1)
			inv_y = scaler.inverse_transform(inv_y)
			inv_y = inv_y[:,0]
			#print(inv_y)
			# invert scaling for forecast/ appending same colums from test to not have problem with inverse scale
			inv_yhat = concatenate((yhat.reshape(test_X.shape[0],1), test_X_[:, -(n_features-1):]), axis=1)
			inv_yhat = scaler.inverse_transform(inv_yhat)
			inv_yhat = inv_yhat[:,0]
			#print(inv_yhat)
			# calculate RMSE
			rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
			#print('Test RMSE: %.3f' % rmse)
			#lista.append(score*score)
			lista.append(rmse)
			best_local=rmse
			#save first run
			if best == 0:
			  best=best_local
			else:
				if best_local < best:
					best=best_local
		#print(lista)
		#print(best)
		#print(chosen[lista.index(best)])
		#print("*****")
		#save first run
		if best_loop == 0:
			best_loop=best
			winner=chosen[lista.index(best)]
		else:
			if best < best_loop:
				best_loop=best
				winner=chosen[lista.index(best)]
	#print(best_loop)
	#print(winner)
	#print("+++++++++++++++++++++++++++++++++++++++++")
	file1.write(str(winner)+",\n")


file1.close()
