# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This file contains methods for training Random Forest model on the
genterated positive/negative centroids. If You want to trin the model again,
uncomment lines: 16-18, 50-53
"""
from fastai.imports import *
from fastai.tabular import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics

#UNCOMMENT FOR TRAINING
#df_train = pd.read_csv('FB13_centroids.csv', header = None)
#df_test =  pd.read_csv('FB13_test_centroids.csv', header = None)
#df_valid = pd.read_csv('FB13_valid_centroids.csv', header = None)

def train_random_forest(df_train):
    """
    train CBRF on the set of the positive/negative centroids and
    save the trained model.
    """
    X_train,  y_train =  df_train.iloc[:,:-1], df_train.iloc[:,-1]

    random_forest = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    random_forest.fit(X_train, y_train)
    filename = 'rf_model.sav'
    pickle.dump(random_forest, open(filename, 'wb'))


def test_random_forest_model(df_test):
    """
    test CBRF on the set of the positive/negative centroids and
    save the trained model.
    """
    X_test, y_test = df_test.iloc[:,:-1], df_test.iloc[:,-1]
    rf = load_rf_model()
    random_forest_preds = rf.predict(X_test)
    print('The accuracy of the Random Forest model is :\t',metrics.accuracy_score(random_forest_preds,y_test))
  
def load_rf_model():
    """
    restore the saved CBRF model.
    """
    return pickle.load(open('rf_model.sav', 'rb'))

#UNCOMMENT FOR TRAINING
#train_random_forest(df_train)
#test_random_forest_model(df_train)
#test_random_forest_model(df_test)
#test_random_forest_model(df_valid)
