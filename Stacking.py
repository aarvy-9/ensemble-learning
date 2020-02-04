import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking
from sklearn.linear_model import SGDClassifier
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_test, y_pred, model):
    cm = confusion_matrix(y_test, y_pred)
    # Show confusion matrix in a separate window
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix - {}'.format(model))
    plt.colorbar()
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


df = pd.read_csv("diabetes_csv.csv")
df['class'] = df['class'].replace({'tested_negative': 0, 'tested_positive': 1})
y = df[['class']]
X = df.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

models = [KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
          SGDClassifier(),
          RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=100, max_depth=3)]
S_train, S_test = stacking(models, X_train, y_train, X_test,
                           regression=False,
                           mode='oof_pred_bag',
                           needs_proba=False,
                           save_dir=None,
                           metric=accuracy_score,
                           n_folds=4,
                           stratified=True,
                           shuffle=True,
                           random_state=0,
                           verbose=2)

model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                      n_estimators=100, max_depth=3)
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)
plot_confusion_matrix(y_test, y_pred, 'XGB')
print('Final prediction score: ', accuracy_score(y_test, y_pred))