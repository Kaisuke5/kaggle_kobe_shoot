import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import model,model2
import argparse
from sklearn import svm
from xgboost.sklearn import XGBClassifier
import scipy as sp
def logloss(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll
def makesubmission(predict_y, savename="submission99.csv"):
	submit_df = pd.read_csv('../data/' + "sample_submission.csv")
	submit_df["shot_made_flag"] = predict_y
	submit_df = submit_df.fillna(np.nanmean(predict_y))
	submit_df.to_csv(savename, index=False)

def factorize(data):
	return (data - np.mean(data,axis = 0)) / np.std(data,axis = 0)


parser = argparse.ArgumentParser(description='kaggle kobe')
parser.add_argument('--gpu', '-g', default=-1, type=int,help='gpu -1')
parser.add_argument('--units', '-u', default=0, type=int,help='units')
parser.add_argument('--epoch', '-n', default=0, type=int,help='epoch')
parser.add_argument('--batchsize', '-b', default=0, type=int,help='batchsize')
parser.add_argument('--train', '-t', default=0, type=int,help='practice')



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


clf = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=550, subsample=0.5, colsample_bytree=0.5, seed=0)
clf.fit(train_x, train_y)
test_y= clf.predict_proba(test_x)[:, 1]
makesubmission(test_y)

print 'done'
# sn = model2.degit_network(units=args.units,gpu=args.gpu)
sn = model.shoot_network(units=args.units,gpu=args.gpu)



if args.train > 0:
	print 'predict validation test set'
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)

	clf = svm.SVC()
	clf.fit(X_train, y_train)
	pred = clf.fit(X_test)

	print 'svm logloss',sn.logloss(y_test,pred)

	sn.fit(X_train,y_train,n_epoch=args.epoch,batchsize=args.batchsize,save=False)
	pred = sn.predict(X_test)
	print 'logloss of test set:',sn.logloss(y_test,pred)

sn.fit(train_x,train_y,n_epoch=args.epoch,batchsize=args.batchsize,save=False)
ans = sn.predict(test_x)
count  = 0


test_data = data[pd.isnull(data_x.shot_made_flag)]
print ans.shape
for i,row in test_data.iterrows():
	#print count,i
	result = ans[count]
	output.write(str(row['shot_id'])+","+str(result)+'\n')
	count += 1

output.close()


