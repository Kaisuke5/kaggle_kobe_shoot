import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import model
import argparse


def factorize(data):
	return (data - np.mean(data,axis = 0)) / np.std(data,axis = 0)


parser = argparse.ArgumentParser(description='kaggle kobe')
parser.add_argument('--gpu', '-g', default=-1, type=int,help='gpu -1')
args = parser.parse_args()

N = 30696
M = 5000

data = pd.read_csv('../data/data.csv')
output = open('output.csv', 'w')

output.write('shot_id,shot_made_flag\n')


train_lst = ['shot_id','lat','loc_x','loc_y','lon','minutes_remaining','playoffs','seconds_remaining','shot_distance','period','shot_made_flag']


lst = ['lat','loc_x','loc_y','lon','minutes_remaining','playoffs','seconds_remaining','period','shot_distance','shot_made_flag']
cols = ['action_type','shot_type','shot_zone_basic','shot_zone_area','shot_zone_range', 'season','opponent']
# lst = ['playoffs','seconds_remaining','shot_distance']


data_x = data[lst]


for col in cols:
	d = pd.get_dummies(data[col])
	data_x = pd.concat((data_x,d),axis=1)


data_x = factorize(data_x)
train_x = data_x[-pd.isnull(data_x.shot_made_flag)]
test_x = data_x[pd.isnull(data_x.shot_made_flag)]
del test_x['shot_made_flag']
del train_x['shot_made_flag']
test_x = test_x.values
train_x = train_x.values
train_y = data[-pd.isnull(data_x.shot_made_flag)]['shot_made_flag'].values



n_epoch = [50,100,150,200,250,300]
units = [500,1000,1200,1500,2000]
batchsize = [100,500,1000]

# #
# n_epoch = [1,2]
# units = [1,2]
# batchsize = [100,500]





file = open('validation.csv','w')
file.write('error,units,batchsize,n_epoch\n')
file.close()
for u in units:
	for b in batchsize:
		for n in n_epoch:
			error = 0
			for i in range(5):
				X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size=0.4, random_state=0)
				sn = model.shoot_network(units=u,gpu=args.gpu)
				sn.fit(X_train,y_train,n_epoch=n,batchsize=b)
				result = sn.predict(X_test)[:,0]
				ans = np.sum((result - y_test) * (result - y_test))
				error += ans
			s = '%5.2f,%d,%d,%d\n' % (error/5,u,b,n)
			file = open('validation.csv','a')
			file.write(s)

file.close()
