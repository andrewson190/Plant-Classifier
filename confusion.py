from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import numpy as np

datasets = ["one-hundred-plants-texture", "one-hundred-plants-margin", "one-hundred-plants-shape"]
X_all, y_all = [], []

for dataset_name in datasets:
    dataset = fetch_openml(dataset_name)
    X = dataset.data
    y = dataset.target
    X_all.append(X[:1599])
    y_all.append(y)

X_combined = np.concatenate(X_all, axis=1)
y_combined = y_all[0]

knn_model = KNeighborsClassifier(n_neighbors=2, weights='distance')
knn_model.fit(X_combined, y_combined)

y_pred = knn_model.predict(X_combined)
print(classification_report(y_combined, y_pred))
print(np.mean(knn_model.score(X_combined, y_combined)))

disp = ConfusionMatrixDisplay.from_estimator(
    knn_model,
    X_combined,
    y_combined,
    cmap=plt.cm.Blues,
    normalize='true',
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.imshow(disp.confusion_matrix)
plt.legend()
plt.show()

