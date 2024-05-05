from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


D1 = fetch_openml("one-hundred-plants-shape", as_frame=False, parser="auto")
D2 = fetch_openml("one-hundred-plants-margin", as_frame=False, parser="auto")
D3 = fetch_openml("one-hundred-plants-texture", as_frame=False, parser="auto")

test = input("which test (1, 2): ")

if (test == "1"):
    for n in range(25, 75, 5):
        print(f"testing on n {n}")
        clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(n), max_iter=100000)
# clf.fit(D3.data, D3.target)
        scores = cross_val_score(clf, D3.data, D3.target)
        print(f"mean score: {scores.mean()} with n {n}")
elif (test == "2"):
    n = 75
    clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(n), max_iter=100000)
# clf.fit(D3.data, D3.target)
    scores = cross_val_score(clf, D3.data, D3.target)
    print(f"mean score: {scores.mean()} with n {n}")

