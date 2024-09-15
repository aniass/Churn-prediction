import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def plot_roc_curve(model, X_test, y_test):
    """The function to plot roc curve"""
    y_pred_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--' )
    plt.plot(fpr, tpr, label='ROC' ,color = 'red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve',fontsize=16)
    plt.legend()
    plt.show()
 
    
def plot_conf_matrix(pred_set, y_test):
    """The function to plot confusion matrix"""
    plt.figure(figsize=(6,4))
    ax = sns.heatmap(confusion_matrix(y_test, pred_set),
                annot=True,fmt = "0.1f",linecolor="k",linewidths=3)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))   

    plt.title("Confusion Matrix",fontsize=14)
    plt.show()
    

def evaluation(model, X_train, y_train, X_test, y_test):
    '''Accuracy score and ROC AUC score calculation; ROC curve and confusion matrix plots'''
    # Accuracy score
    acc = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    acc_score = round(acc.mean(), 2)
    print('Accuracy score: %s' % acc_score)
    # ROC AUC score
    pred_prob = model.predict_proba(X_test)
    score = roc_auc_score(y_test, pred_prob[:,1])
    roc_score = round(score, 2)
    print('ROC AUC score: %s' % roc_score)
    pred_y = model.predict(X_test)
    return pred_y 
    