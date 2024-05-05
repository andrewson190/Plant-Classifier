from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import cross_val_score

datasets = ["one-hundred-plants-texture", "one-hundred-plants-margin", "one-hundred-plants-shape"]
X_all, y_all = [], []

for dataset_name in datasets:
    dataset = fetch_openml(dataset_name)
    X = dataset.data
    y = dataset.target
    X_all.append(X[:1599])
    y_all.append(y )

X_combined = np.concatenate(X_all, axis=1)
y_combined = y_all[0]

knn_model = KNeighborsClassifier(n_neighbors=5,  weights='distance')

cv_scores = cross_val_score(knn_model, X_combined, y_combined, cv=5)

mean_cv_score = np.mean(cv_scores)
print("Out-sample:", mean_cv_score)

knn_model.fit(X_combined, y_combined)
print("In-sample", np.mean(knn_model.score(X_combined, y_combined)))


