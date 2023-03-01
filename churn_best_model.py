import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from helper_functions import plot_roc_curve, plot_conf_matrix
from joblib import dump


URL = 'data\Churn_Modelling.csv'


def clean_data(df):
    '''Cleaning data and convert non numeric values'''
    data = df.drop(['CustomerId','Surname','RowNumber'],axis=1)
    data['Gender'] = data['Gender'].map({'Male' : 0, 'Female' : 1})
    data = pd.get_dummies(data, columns = ['Geography'])
    return data


def read_data(path):
    '''Read and preprocess data'''
    data = pd.read_csv(path)
    df = clean_data(data)
    return df


def splitting_data(data):
    '''Spliting data into train and test set'''
    X = data.drop('Exited', axis=1)
    Y = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10, stratify=Y)
    return X_train, X_test, y_train, y_test

  
def evaluation(model):
    '''Accuracy score and roc auc score calculation,
       roc curve and confusion matrix plots'''
    # accuracy score
    acc = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    acc_score = round(acc.mean(), 2)
    # roc auc score
    pred_prob = model.predict_proba(X_test)
    score = roc_auc_score(y_test, pred_prob[:,1])
    roc_score = round(score, 2)
    pred_y = model.predict(X_test)
    print('Accuracy score: %s' % acc_score)
    print('ROC AUC score: %s' % roc_score)
    # plots
    print(plot_roc_curve(model, X_test, y_test))
    print(plot_conf_matrix(pred_y, y_test))


def train_models(X_train, X_test, y_train, y_test):
    ''' Calculating models with score'''
    model = Pipeline(steps=[('scaler', StandardScaler()),
                            ('classifier', RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0))])
    model.fit(X_train, y_train)
    scores = evaluation(model)
    return scores

  
if __name__ == '__main__':
    df = read_data(URL)
    X_train, X_test, y_train, y_test = splitting_data(df)
    model = train_models(X_train, X_test, y_train, y_test)
    # save the model
    dump(model, 'models/rf_model.pkl')
