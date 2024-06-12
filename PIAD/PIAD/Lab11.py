from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

X, y = make_classification(n_samples=2000, n_features=4, n_classes=4, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

classifiers = [
    (OneVsRestClassifier(SVC()), 'OvR SVC'),
    (OneVsOneClassifier(Perceptron()), 'OvO Perceptron'),
    (OneVsOneClassifier(SVC()), 'OvO SVC'),
    (OneVsOneClassifier(SVC(C=1.0)), 'OvO SVC(C=1.0)'),
    (OneVsRestClassifier(Perceptron()), 'OvR Perceptron'),
    (OneVsRestClassifier(LogisticRegression()), 'OvR Logistic Regression'),
    (OneVsRestClassifier(SVC(C=1.0)), 'OvR SVC(C=1.0)'),
    (OneVsOneClassifier(LogisticRegression()), 'OvO Logistic Regression')
]

for clf, clf_name in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    if hasattr(clf.estimators_[0], "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)
    auc_score = roc_auc_score(np.eye(4)[y_test], y_score, average='macro')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(np.eye(4)[y_test][:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(np.eye(4)[y_test][:, i], y_score[:, i])
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f'Classification Results for {clf_name}\nAccuracy: {accuracy:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}, AUC: {auc_score:.2f}')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='.')
    plt.scatter(X_test[y_pred != y_test, 0], X_test[y_pred != y_test, 1], c='red', marker='x', label='Misclassified')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(f'ROC Curve for {clf_name}')
    for i in range(4):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

X, y = make_classification(n_samples=1800, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1, edgecolor='k')
plt.title('Oczekiwane')

plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Set1, edgecolor='k')
plt.title('Obliczone')

plt.subplot(1, 3, 3)
correct = y_test == y_pred
plt.scatter(X_test[:, 0], X_test[:, 1], c=correct, cmap=plt.cm.Paired, edgecolor='k')
plt.title('Różnice')

plt.show()

X, y = make_classification(n_samples=1800, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


clf = LogisticRegression()
clf.fit(X_train, y_train)

plt.figure(figsize=(8, 6))
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Przykładowa wizualizacja krzywych dyskryminacyjnych i podprzestrzeni związanych z klasami')
plt.show()


X, y = make_classification(n_samples=1800, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

classifiers = [
    (OneVsRestClassifier(SVC()), 'OvR SVC'),
    (OneVsOneClassifier(Perceptron()), 'OvO Perceptron'),
    (OneVsOneClassifier(SVC()), 'OvO SVC'),
    (OneVsOneClassifier(SVC(C=1.0)), 'OvO SVC(C=1.0)'),
    (OneVsRestClassifier(Perceptron()), 'OvR Perceptron'),
    (OneVsRestClassifier(LogisticRegression()), 'OvR Logistic Regression'),
    (OneVsRestClassifier(SVC(C=1.0)), 'OvR SVC(C=1.0)'),
    (OneVsOneClassifier(LogisticRegression()), 'OvO Logistic Regression')
]

results = []
for clf, name in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), average='weighted', multi_class='ovr')
    results.append((name, accuracy, recall, precision, f1, auc))
labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC AUC']
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(labels))
width = 0.1

for i, (name, accuracy, recall, precision, f1, auc) in enumerate(results):
    ax.bar(x + i * width, [accuracy, recall, precision, f1, auc], width, label=name)

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Classification Quality Metrics')
ax.set_xticks(x + width * (len(classifiers) - 1) / 2)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

