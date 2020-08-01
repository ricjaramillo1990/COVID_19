# -*- coding: utf-8 -*-
"""Multivariate_Tuning"""

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
import sys
from datetime import datetime
from keras.optimizers import Adam, Nadam, Adamax
import pandas as pd
from pandas import DataFrame
from pandas import concat
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

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


fecha=datetime.today().strftime('%Y%m%d')
total_start_time = time.time()

#SEED
numpy.random.seed(7)
tf.random.set_seed(123)

#SELECTING COUNTRY
country=['Canada','Denmark','France','Germany','Netherlands','United Kingdom','Italy','Israel','Australia','Luxembourg','Japan','Portugal','Croatia','Finland','Ireland','Iceland','Norway','New Zealand','Brazil','Mexico','United States','India','Russia','Poland','Sweden']
steps=[5,7,9]


if sys.argv[1] == 'deaths':
	ext_variable='ConfirmedDeaths'
	file1 = open("/home/org00004/LSTM/"+fecha+"_multivdeaths_light.txt","w")
elif sys.argv[1] == 'confirmed':
	ext_variable='ConfirmedCases'
	file1 = open("/home/org00004/LSTM/"+fecha+"_multiconfirm_light.txt","w")

#LOADING DATASET
dataset_global = pd.read_csv('https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv')
dataset_global["Date"]= pd.to_datetime(dataset_global["Date"], format='%Y%m%d')

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


for i in country:
	lista=[]
	best=0
	for w in steps:
		dataset=dataset_global.loc[( (dataset_global.CountryName ==i) & (dataset_global.Date > "2020-03-01")),['CountryName','CountryCode','Date','C1_School closing','C1_Flag','C2_Workplace closing','C2_Flag','C3_Cancel public events','C3_Flag','C4_Restrictions on gatherings','C4_Flag','C5_Close public transport','C5_Flag','C6_Stay at home requirements','C6_Flag','C7_Restrictions on internal movement','C7_Flag','C8_International travel controls','E1_Income support','E1_Flag','E2_Debt/contract relief','E3_Fiscal measures','E4_International support','H1_Public information campaigns','H1_Flag','H2_Testing policy','H3_Contact tracing','H4_Emergency investment in healthcare','H5_Investment in vaccines','M1_Wildcard','ConfirmedCases','ConfirmedDeaths']].set_index('Date')
		dataset.columns = ['CountryName','CountryCode','C1','C1_Flag','C2','C2_Flag','C3','C3_Flag','C4','C4_Flag','C5','C5_Flag','C6','C6_Flag','C7','C7_Flag','C8','E1','E1_Flag','E2','E3','E4','H1','H1_Flag','H2','H3','H4','H5','M1','ConfirmedCases','ConfirmedDeaths']

		#Filling NaN due to uplead delay (Because data are updated on twice-weekly cycles, but not every country is updated in every cycle)
		dataset.fillna(method='ffill', inplace=True)
		#columns to look for missing values in deaths counts.
		dataset.dropna(subset=[ext_variable])

		dataset['C1_new']=round(100*((dataset['C1']-(0.5*(c1['F']-dataset['C1_Flag'])))/c1['N']),2)
		#dataset['C1_new'][dataset['C1_new'] < 0] = 0
		dataset.loc[(dataset['C1_new'] < 0), 'C1_new'] = 0
		dataset['C1_new']=dataset['C1_new'].fillna(0)

		dataset['C2_new']=round(100*((dataset['C2']-(0.5*(c2['F']-dataset['C2_Flag'])))/c2['N']),2)
		#dataset['C2_new'][dataset['C2_new'] < 0] = 0
		dataset.loc[(dataset['C2_new'] < 0), 'C2_new'] = 0
		dataset['C2_new']=dataset['C2_new'].fillna(0)

		dataset['C3_new']=round(100*((dataset['C3']-(0.5*(c3['F']-dataset['C3_Flag'])))/c3['N']),2)
		#dataset['C3_new'][dataset['C3_new'] < 0] = 0
		dataset.loc[(dataset['C3_new'] < 0), 'C3_new'] = 0
		dataset['C3_new']=dataset['C3_new'].fillna(0)

		dataset['C4_new']=round(100*((dataset['C4']-(0.5*(c4['F']-dataset['C4_Flag'])))/c4['N']),2)
		#dataset['C4_new'][dataset['C4_new'] < 0] = 0
		dataset.loc[(dataset['C4_new'] < 0), 'C4_new'] = 0
		dataset['C4_new']=dataset['C4_new'].fillna(0)

		dataset['C5_new']=round(100*((dataset['C5']-(0.5*(c5['F']-dataset['C5_Flag'])))/c5['N']),2)
		#dataset['C5_new'][dataset['C5_new'] < 0] = 0
		dataset.loc[(dataset['C5_new'] < 0), 'C5_new'] = 0
		dataset['C5_new']=dataset['C5_new'].fillna(0)

		dataset['C6_new']=round(100*((dataset['C6']-(0.5*(c6['F']-dataset['C6_Flag'])))/c6['N']),2)
		#dataset['C6_new'][dataset['C6_new'] < 0] = 0
		dataset.loc[(dataset['C6_new'] < 0), 'C6_new'] = 0
		dataset['C6_new']=dataset['C6_new'].fillna(0)

		dataset['C7_new']=round(100*((dataset['C7']-(0.5*(c7['F']-dataset['C7_Flag'])))/c7['N']),2)
		#dataset['C7_new'][dataset['C7_new'] < 0] = 0
		dataset.loc[(dataset['C7_new'] < 0), 'C7_new'] = 0
		dataset['C7_new']=dataset['C7_new'].fillna(0)

		dataset['C8_new']=round(100*((dataset['C8']-(0.5*(c8['F'])))/c8['N']),2)
		#dataset['C8_new'][dataset['C8_new'] < 0] = 0
		dataset.loc[(dataset['C8_new'] < 0), 'C8_new'] = 0
		dataset['C8_new']=dataset['C8_new'].fillna(0)

		dataset['E1_new']=round(100*((dataset['E1']-(0.5*(e1['F']-dataset['E1_Flag'])))/e1['N']),2)
		#dataset['E1_new'][dataset['E1_new'] < 0] = 0
		dataset.loc[(dataset['E1_new'] < 0), 'E1_new'] = 0
		dataset['E1_new']=dataset['E1_new'].fillna(0)

		dataset['E2_new']=round(100*((dataset['E2']-(0.5*(e2['F'])))/e2['N']),2)
		#dataset['E2_new'][dataset['E2_new'] < 0] = 0
		dataset.loc[(dataset['E2_new'] < 0), 'E2_new'] = 0
		dataset['E2_new']=dataset['E2_new'].fillna(0)

		dataset['H1_new']=round(100*((dataset['H1']-(0.5*(h1['F']-dataset['H1_Flag'])))/h1['N']),2)
		#dataset['H1_new'][dataset['H1_new'] < 0] = 0
		dataset.loc[(dataset['H1_new'] < 0), 'H1_new'] = 0
		dataset['H1_new']=dataset['H1_new'].fillna(0)

		dataset['H2_new']=round(100*((dataset['H2']-(0.5*(h2['F'])))/h2['N']),2)
		#dataset['H2_new'][dataset['H2_new'] < 0] = 0
		dataset.loc[(dataset['H2_new'] < 0), 'H2_new'] = 0
		dataset['H2_new']=dataset['H2_new'].fillna(0)

		dataset['H3_new']=round(100*((dataset['H3']-(0.5*(h3['F'])))/h3['N']),2)
		#dataset['H3_new'][dataset['H3_new'] < 0] = 0
		dataset.loc[(dataset['H3_new'] < 0), 'H3_new'] = 0
		dataset['H3_new']=dataset['H3_new'].fillna(0)

		dataset['politic']=dataset[['C1_new','C2_new','C3_new','C4_new','C5_new','C6_new','C7_new','C8_new']].mean(axis = 1)
		dataset['economic']=dataset[['E1_new','E2_new']].mean(axis = 1)
		dataset['health']=dataset[['H1_new','H2_new','H3_new']].mean(axis = 1)

		#creating death toll per day
		dataset['Day_Count']=dataset[ext_variable].diff()
		dataset=dataset.drop(columns=[ext_variable])
		dataset.loc[(dataset['Day_Count'] < 0), 'Day_Count'] = 0

		#final dataset
		dataset=dataset[['Day_Count','C5_new','E1_new','E2_new']]
		dataset=dataset.dropna()
		values = dataset.values

		# ensure all data is float
		values = values.astype('float32')
		# scaling
		scaler = MinMaxScaler(feature_range=(0, 1))
		scaled = scaler.fit_transform(values)
		# specify the number of time_steps
		time_step = w
		n_features = len(dataset.columns)
		features= time_step*n_features
		# frame as supervised learning
		reframed = series_to_supervised(scaled, time_step, 1)
		#print(reframed.shape)

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

		# CREATE AND FIT LSTM NETWORK
		def create_model(neurons=150,activation='relu', learn_rate=0.01 , opt=Adam):
			model = Sequential()
			model.add(LSTM(units=neurons, activation=activation ,input_shape=(train_X.shape[1], train_X.shape[2])))
			model.add(Dense(units=1))
			#compile
			#The momentum is the first beta_1 (0<beta<1. Generally close to 1) in Nadam/Adam/Adamax  = momentum (float >= 0) in SGD
			lr=learn_rate
			optimizer = opt(learning_rate=lr)
			model.compile(optimizer=optimizer, loss='mean_squared_error' )
			return model


		kears_estimator = KerasRegressor(build_fn=create_model, verbose=0)

		# DEFINING GRID SERACH PARAMETERS
		batch_size=[r for r in range(int(len(dataset)/4),len(dataset),int(len(dataset)/4))][:-1]
		epochs = [300,400,500]
		activation = ['relu','tanh']
		neurons = [50,125,200]
		learn_rate = [0.005, 0.01, 0.05]
		#momentum = [0.8, 0.9]
		opt = [Adam,Nadam]


		param_grid = dict(batch_size=batch_size, epochs=epochs, activation=activation, neurons=neurons,learn_rate=learn_rate , opt=opt)
		#grid = GridSearchCV(estimator=kears_estimator, param_grid=param_grid, n_jobs=-1, cv=3)
		grid = RandomizedSearchCV(estimator=kears_estimator, param_distributions=param_grid, n_jobs=-1, cv=3 ,n_iter=30)
		start_time = time.time()
		grid_result = grid.fit(train_X, train_y)
		end_time = time.time()

		asd=[grid_result.best_params_[i] for i in grid_result.best_params_]
		asd.insert(0,w)
		best_local=grid_result.best_score_*grid_result.best_score_

		if str(asd[1]) =="<class 'keras.optimizers.Adam'>":
			asd[1]=0
		elif str(asd[1]) =="<class 'keras.optimizers.Nadam'>":
  			asd[1]=1
		elif str(asd[1]) =="<class 'keras.optimizers.Adamax'>":
			asd[1]=2

		if w == steps[0]:
			lista=asd
			best=best_local
		else:
			if best_local < best:
				best=best_local
				lista=asd
			if w == steps[-1]:
				file1.write("\""+i+"\":"+str(lista).replace("'", "\"")+",\n")
				#print(w)

total_final_time = time.time()
#print("Total Execution time: " + str(round((total_final_time - total_start_time)/60,1)) + ' min')
file1.close()
