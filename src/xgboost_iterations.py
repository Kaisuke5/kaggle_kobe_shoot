import model_factory as mf
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import  XGBClassifier
import random
import model_factory as mf
import pandas as pd
import models.data as data
import numpy as np

def makesubmission(predict_y, savename="submission99.csv"):
    submit_df = pd.read_csv('../data/'+"sample_submission.csv")
    submit_df["shot_made_flag"] = predict_y
    submit_df = submit_df.fillna(np.nanmean(predict_y))
    submit_df.to_csv(savename, index=False)


if __name__ == "__main__":
    N = 1
    DIR = 'trained_model/'
    datafile = 'data.csv'
    df = pd.read_csv('../data/'+ datafile)
    df = data.preproc(df)
    train_x,train_y,test_x = mf.make_data(df)
    ans = np.zeros(5000)
    for i in range(N):
        xgb = XGBClassifier(max_depth=random.randint(4,8), learning_rate=random.uniform(0.01,0.05), n_estimators=random.randint(300,700), subsample=0.5, colsample_bytree=0.5, seed=0)
        xgb.fit(train_x,train_y)
        result = xgb.predict_proba(test_x)[:,1]
        ans +=  result

    makesubmission(ans/N,savename='iter.csv')
