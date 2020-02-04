import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import itertools
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt


def get_data():
    df = pd.read_csv("diabetes_csv.csv")
    df['class'] = df['class'].replace({'tested_negative': 0, 'tested_positive': 1})
    y = df[['class']]
    X = df.iloc[:, :-1]
    return X, y


def plot_roc_curve(fpr, tpr, model_name):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - {}'.format(model_name))
    plt.legend()
    plt.show()


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


def cascade_models(models, X_train_values, y_train_values, test=False):
    for i, model in enumerate(models):
        if not test:
            model[1].fit(X_train_values, y_train_values.values.ravel())
        if i != len(models) - 1:
            X_train_values.insert(X_train_values.shape[1], "T{}".format(i), model[1].predict(X_train_values), True)
    return model


if __name__ == "__main__":
    max_acc = 0
    best_seq = None
    train_models = [
        ("KNN", KNeighborsClassifier(n_neighbors=5,
                             n_jobs=-1)),
        ("SGD", SGDClassifier(loss='log')),
        ("RFC", RandomForestClassifier(random_state=0, n_jobs=-1,
                               n_estimators=100, max_depth=3))
    ]
    for each in list(itertools.permutations(train_models)):
        X, y = get_data()
        seq = ""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        model = cascade_models(each, X_train, y_train)
        x_pred = model[1].predict(X_train)
        train_acc = accuracy_score(y_train, x_pred)
        for count, i in enumerate(each):
            seq += i[0]
            if count != len(each) - 1:
                seq += " -> "
        print(seq)
        print("Train accuracy: ", train_acc)
        model = cascade_models(each, X_test, y_test, test=True)
        y_pred = model[1].predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred, seq)
        if model[0] != "SGD":
            probs = model[1].predict_proba(X_test)
            probs = probs[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, probs)
            plot_roc_curve(fpr, tpr, seq)
        print("Test accuracy: ", test_acc)
        print("------------------------------")

        if test_acc > max_acc:
            max_acc = test_acc
            best_seq = [i[0] for i in each]
    print('Max axxuracy observed : ', max_acc)
    print('Sequence with maximum accuracy : ', best_seq)