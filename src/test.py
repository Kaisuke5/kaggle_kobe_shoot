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
parser.add_argument('--units', '-u', default=0, type=int,help='units')
parser.add_argument('--epoch', '-n', default=0, type=int,help='epoch')
parser.add_argument('--batchsize', '-b', default=0, type=int,help='batchsize')



args = parser.parse_args()

N = 30696
M = 5000

data = pd.read_csv('../data/data.csv')
output = open('output.csv', 'w')

output.write('shot_id,shot_made_flag\n')

lst = ['lat','loc_x','loc_y','lon','minutes_remaining','playoffs','seconds_remaining','period','shot_distance','shot_made_flag']
cols = ['action_type','shot_type','shot_zone_basic','shot_zone_area','shot_zone_range', 'season','opponent']
# lst = ['playoffs','seconds_remaining','shot_distance']

data_x = data[lst]

for col in cols:
	d = pd.get_dummies(data[col])
	data_x = pd.concat((data_x,d),axis=1)

data_x = factorize(data_x)
train_data = data_x[-pd.isnull(data_x.shot_made_flag)]
test_data = data_x[pd.isnull(data_x.shot_made_flag)]
del train_data['shot_made_flag']
del test_data['shot_made_flag']
test_x = test_data.values
train_x = train_data.values
train_y = data[-pd.isnull(data_x.shot_made_flag)]['shot_made_flag'].values



sn = model.shoot_network(units=args.units,gpu=args.gpu)
sn.fit(train_x,train_y,n_epoch=args.epoch,batchsize=args.batchsize,save=True)
ans = sn.predict(test_x)
count  = 0


test_data = data[pd.isnull(data_x.shot_made_flag)]
print ans.shape
for i,row in test_data.iterrows():
	#print count,i
	result = ans[count][0]
	if result > 1: result = 1.0
	elif result < 0: result = 0.0
	output.write(str(row['shot_id'])+","+str(result)+'\n')
	count += 1

output.close()


