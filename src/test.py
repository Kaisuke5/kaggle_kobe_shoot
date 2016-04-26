import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import model
import argparse


def factorize(data):
	return (data - np.mean(data,axis = 0)) / np.var(data,axis = 0)


parser = argparse.ArgumentParser(description='kaggle kobe')
parser.add_argument('--gpu', '-g', default=-1, type=int,help='gpu -1')
args = parser.parse_args()

N = 30696
M = 5000

data = pd.read_csv('../data/data.csv')
output = open('output.csv', 'w')

output.write('shot_id,shot_made_flag\n')


train_lst = ['shot_id','lat','loc_x','loc_y','lon','minutes_remaining','playoffs','seconds_remaining','shot_distance','shot_made_flag']
lst = ['lat','loc_x','loc_y','lon','minutes_remaining','playoffs','seconds_remaining','shot_distance','shot_made_flag']
cols = ['combined_shot_type', 'shot_zone_range', 'season','period']
# lst = ['playoffs','seconds_remaining','shot_distance']


data_x = data[lst]


#label to binary features
# data_sparse = data['shot_id']


for col in cols:
	d = pd.get_dummies(data[col])
	data_x = pd.concat((data_x,d),axis=1)


# data_sparse.drop('shot_id', axis=1)
# U,s,V = np.linalg.svd(data_sparse.values,full_matrices=False)
# d = 8
# svd_features = np.dot(U[:,0:d],np.dot(np.diag(s[:d]),V.T[:,0:d].T))
# print svd_features
#


data_x = factorize(data_x)
train_x = data_x[-pd.isnull(data_x.shot_made_flag)]
test_x = data_x[pd.isnull(data_x.shot_made_flag)]
del test_x['shot_made_flag']
del train_x['shot_made_flag']
test_x = test_x.values
train_x = train_x.values
train_y = data[-pd.isnull(data_x.shot_made_flag)]['shot_made_flag'].values


sn = model.shoot_network(units=1200,gpu=args.gpu)
sn.fit(train_x,train_y,n_epoch=200,batchsize=1000,save=True)

ans = sn.predict(test_x)
count  = 0
test_data = data[pd.isnull(data.shot_made_flag)]
for i,row in test_data.iterrows():
	#print count,i
	result = ans[count][0]
	if result > 1: result = 1
	elif result < 0: result = 0
	output.write(str(row['shot_id'])+","+str(result)+'\n')
	count += 1

output.close()


