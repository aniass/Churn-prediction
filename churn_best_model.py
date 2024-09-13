import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from helper_functions import plot_roc_curve, plot_conf_matrix
from joblib import dump


URL = 'C:\Python Scripts\Datasets\churn\Churn_Modelling.csv'

RANDOM_STATE = 0
N_ESTIMATORS = 200


def clean_data(df):
    '''Cleaning data and convert non numeric values'''
    data = df.drop(['CustomerId','Surname','RowNumber'],axis=1)
    data['Gender'] = data['Gender'].map({'Male' : 0, 'Female' : 1})
    data = pd.get_dummies(data, columns = ['Geography'])
    return data


def read_data(path):
    '''Read and preprocess data'''
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    df = clean_data(data)
    return df


def splitting_data(data, test_size=0.2):
    '''Spliting data into train and test set'''
    X = data.drop('Exited', axis=1)
    Y = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=RANDOM_STATE, stratify=Y)
    return X_train, X_test, y_train, y_test

  
def evaluation(model, X_test, y_test):
    '''Accuracy score and ROC AUC score calculation; ROC curve and confusion matrix plots'''
    # Accuracy score
    acc = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    acc_score = round(acc.mean(), 2)
    print('Accuracy score: %s' % acc_score)
    # ROC AUC score
    pred_prob = model.predict_proba(X_test)
    score = roc_auc_score(y_test, pred_prob[:,1])
    roc_score = round(score, 2)
    pred_y = model.predict(X_test)
    print('ROC AUC score: %s' % roc_score)
    print(plot_conf_matrix(pred_y, y_test))


def train_model(X_train, X_test):
    '''Calculating model with score'''
    model = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=N_ESTIMATORS, criterion='entropy', random_state=RANDOM_STATE))])
    model.fit(X_train, y_train)
    return model

  
if __name__ == '__main__':
    df = read_data(URL)
    if df is not None:
        X_train, X_test, y_train, y_test = splitting_data(df)
        model = train_model(X_train, y_train)
        # Save the model 
        #dump(model, 'models/rf_model.pkl')
        # Evaluate the model
        evaluation(model, X_test, y_test)
        # Plots
        print(plot_roc_curve(model, X_test, y_test))
