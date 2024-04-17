import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib as plt

def fit_rfe_model(x,y):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    cv = StratifiedKFold(n_splits=3)
    rfe = RFECV(estimator=model, step=10, cv=cv, scoring='accuracy', min_features_to_select=20)
    rfe.fit(x_scaled, y)
    return rfe

x = alligned_single_transcriptome
y = vae_metadata['GT']

rfe = fit_rfe_model(x,y)

print("Optimal number of features : %d" % rfe.n_features_)

cv_scores = rfe.cv_results_['mean_test_score']  

def plot_features_against_cv(cv_scores):
    # Plotting the number of features against cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validation score (mean of accuracy)")
    plt.plot(range(1, len(cv_scores) + 1), cv_scores)
    plt.title("Feature Selection Performance")
    plt.show()

selected_features = pd.DataFrame({'Feature': list(X.columns), 'Importance': rfe.support_})

