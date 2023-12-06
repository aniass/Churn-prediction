import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
# Model packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier


URL = r'\data\Churn_Modelling.csv'
N_NEIGHBORS = 20
N_ESTIMATORS = 200
RANDOM_STATE = 0


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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y)
    return X_train, X_test, y_train, y_test


def acc_score(model):
    """Accuracy score calculation from cross validation"""
    acc = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    score = round(acc.mean(), 2)
    return score


def roc_score(model):
    """Roc auc score calculation"""
    pred_prob = model.predict_proba(X_test)
    score = roc_auc_score(y_test, pred_prob[:,1])
    auc_score = round(score, 2)
    return auc_score

   
def train_models(X_train, X_test, y_train, y_test):
    ''' Calculating models with score'''
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(random_state=RANDOM_STATE),
        RandomForestClassifier(n_estimators=N_ESTIMATORS, criterion='entropy', random_state=RANDOM_STATE),
        KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric="minkowski", p=2),
        AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
        XGBClassifier(random_state=RANDOM_STATE)]
     
    for classifier in classifiers:
        model = Pipeline(steps=[('scaler', StandardScaler()),
                                ('classifier', classifier)])
        model.fit(X_train, y_train)
        
        score = roc_score(model)
        score_acc = acc_score(model)
        param_dict = {
                     'Model': classifier.__class__.__name__,
                     'Roc Auc': score,
                     'Accuracy score': score_acc
                     }
        models = models.append(pd.DataFrame(param_dict, index=[0]))
        
    models.reset_index(drop=True, inplace=True)
    models_sorted = models.sort_values(by='Roc Auc', ascending=False)
    return models_sorted
     
  
if __name__ == '__main__':
    df = read_data(URL)
    X_train, X_test, y_train, y_test = splitting_data(df)
    result_models = train_models(X_train, X_test, y_train, y_test)
    print(result_models)
