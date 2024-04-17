import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

names = [
    "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    "Naive Bayes", "QDA"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def run_k_fold_cross_validation(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {name: {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []} for name in names}

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        fold_results = []

        for name, clf in zip(names, classifiers):
            model = clf.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            
            metrics[name]['accuracy'].append(acc)
            metrics[name]['f1_score'].append(f1)
            metrics[name]['precision'].append(prec)
            metrics[name]['recall'].append(rec)

            fold_results.append(y_pred)

        # Compute Cohen's Kappa for this fold (comparing all pairs of classifiers)
        kappa_matrix = np.zeros((len(classifiers), len(classifiers)))
        for i in range(len(fold_results)):
            for j in range(i + 1, len(fold_results)):
                kappa = cohen_kappa_score(fold_results[i], fold_results[j])
                kappa_matrix[i, j] = kappa
                kappa_matrix[j, i] = kappa
        avg_kappa = np.mean(kappa_matrix[np.triu_indices_from(kappa_matrix, 1)])
        print(f"Average Cohen's Kappa for fold {fold+1}: {avg_kappa:.2f}")

        
    for name in names:

        avg_cm = np.mean(metrics[name]['conf_matrix'], axis=0).astype(int)
        disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm, display_labels=np.unique(y))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Average Confusion Matrix for {name}')
        plt.show()

        print(f"{name} - Average Metrics:")
        print(f"Accuracy: {np.mean(metrics[name]['accuracy']):.2f}")
        print(f"F1 Score: {np.mean(metrics[name]['f1_score']):.2f}")
        print(f"Precision: {np.mean(metrics[name]['precision']):.2f}")
        print(f"Recall: {np.mean(metrics[name]['recall']):.2f}")
        print("---------------------------------------")

    # Plotting validation accuracy across folds for each classifier
    plt.figure(figsize=(12, 6))
    for name in names:
        plt.plot(metrics[name]['accuracy'], label=f'{name} Acc')
    plt.title('Validation Accuracy by Fold')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return metrics

# Example dataset loading
data = pd.read_csv('your_data.csv')
X = data.drop(['label', 'time_to_event', 'event_occurred'], axis=1).values
y = data['label'].values

# Running the cross-validation
results = run_k_fold_cross_validation(X, y)

# Assuming survival analysis data is part of the same dataset
perform_kaplan_meier_analysis(data)

