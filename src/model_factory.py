from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor,ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
import pickle
import pandas as pd
import models.data as data
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
import scipy as sp
import numpy as np
DIR = 'trained_model/'
datafile = 'data.csv'
df = pd.read_csv('../data/'+ datafile)
df = data.preproc(df)

def test_it(data):
    clf = RandomForestClassifier(n_jobs=-1)  # A super simple classifier
    return cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                           scoring='roc_auc', cv=10
                          )
def make_data(df):
    x_mapper, y_mapper = data.mapper(df)
    train_df, test_df = data.split(df)
    train_x_vec = x_mapper.transform(train_df.copy())
    train_y_vec = y_mapper.transform(train_df.copy())
    test_x_vec = x_mapper.transform(test_df.copy())
    return train_x_vec,train_y_vec,test_x_vec


def train_save(model,df,save_name):
    train_x_vec, train_y_vec, test_x_vec = make_data(df)
    model.fit(train_x_vec, train_y_vec)
    with open(save_name,'w') as f:
        pickle.dump(model,f)
    return train_x_vec,train_y_vec

def logloss(act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        pred[pred >= 1] = 0.9999999
        ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
        ll = ll * -1.0/len(act)
        return ll


def test(model,df):
    N = 3
    loss = 0.0
    train_x,train_y,test_x = make_data(df)
    for i in range(N):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size=0.4, random_state=0)
        pred = model.predict_proba(X_test)[:, 1]
        loss += logloss(y_test[:,0],pred)

    return loss / N


def make_features_from_multimodels(models,train_x,test_x):
    Train_X = train_x
    Test_X = test_x
    for m in models:
        x = m.predict_proba(train_x)
        xx = m.predict_proba(test_x)
        Train_X = np.c_[Train_X,x]
        Test_X = np.c_[Test_X, xx]

    return Train_X,Test_X



def makesubmission(predict_y, savename="submission99.csv"):
    submit_df = pd.read_csv('../data/'+"sample_submission.csv")
    submit_df["shot_made_flag"] = predict_y
    submit_df = submit_df.fillna(np.nanmean(predict_y))
    submit_df.to_csv(savename, index=False)


if __name__=='__main__':
    xgboost = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=550, subsample=0.5, colsample_bytree=0.5, seed=0)
    rf = RandomForestClassifier()
    models = [xgboost,rf]
    models_name = ['xgboost','rf']
    x,y,xx = make_data(df)

    # for model, name in zip(models,models_name):
    #     print 'train %s' % name
    #     train_save(model,df,DIR + name +'.dump')
    f = open('trained_model/xgboost.dump','r')
    xg = pickle.load(f)
    f = open('trained_model/rf.dump','r')
    rf = pickle.load(f)

    train_x,test_x = make_features_from_multimodels([xg,rf],x,xx)
    print train_x.shape,x.shape,xx.shape,test_x
    clf = LogisticRegression()
    clf.fit(train_x,y)
    # print test(clf,df)
    result = clf.predict_proba(test_x)
    print result[:,1]
    makesubmission(result,'boost.csv')