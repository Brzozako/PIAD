import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import time

def generate_data():
    X, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2,
                               n_classes=2, random_state=42)
    return X, y

def visualize_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title("Przykładowa wizualizacja zbioru danych")
    plt.show()

def evaluate_classifier(name, clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plt.scatter(X_test[:, 0], X_test[:, 1], c=(y_test == y_pred), cmap='coolwarm', marker='.')
    plt.title(f"Błędy klasyfikacji dla {name}")
    plt.show()

    fpr, tpr, _ = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"AUC = {metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"Krzywa ROC dla {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')
    plt.show()

def run_experiment(classifiers):
    results = []

    for name, clf in classifiers.items():
        accuracy_scores = []
        recall_scores = []
        precision_scores = []
        f1_scores = []
        roc_auc_scores = []
        training_times = []
        testing_times = []

        for _ in range(100):
            X, y = generate_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            start_train = time.time()
            clf.fit(X_train, y_train)
            end_train = time.time()

            start_test = time.time()
            y_pred = clf.predict(X_test)
            end_test = time.time()

            accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
            recall_scores.append(metrics.recall_score(y_test, y_pred))
            precision_scores.append(metrics.precision_score(y_test, y_pred))
            f1_scores.append(metrics.f1_score(y_test, y_pred))
            roc_auc_scores.append(metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

            training_times.append(end_train - start_train)
            testing_times.append(end_test - start_test)

        results.append({
            "Classifier": name,
            "Accuracy": np.mean(accuracy_scores),
            "Recall": np.mean(recall_scores),
            "Precision": np.mean(precision_scores),
            "F1 Score": np.mean(f1_scores),
            "ROC AUC": np.mean(roc_auc_scores),
            "Training Time": np.mean(training_times),
            "Testing Time": np.mean(testing_times)
        })

        df_results = pd.DataFrame(results)

    return df_results

def plot_results(df_results):
    df_results.set_index("Classifier").plot(kind='bar', figsize=(15, 10))
    plt.title("Porównanie klasyfikatorów")
    plt.ylabel("Wartość")
    plt.xticks(rotation=45)
    plt.show()

classifiers = {
    "GaussianNB": GaussianNB(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

X, y = generate_data()
visualize_data(X, y)
df_results = run_experiment(classifiers)
plot_results(df_results)
for name, clf in classifiers.items():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    evaluate_classifier(name, clf, X_train, X_test, y_train, y_test)
