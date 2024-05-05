from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from matplotlib.pyplot import plot, show, xlabel, ylabel
from random import choices
from numpy import empty, unique, arange, concatenate

def test(d, loss, name, alpha=.0001):
    M = SGDClassifier(loss=loss, n_jobs=-1, alpha=alpha)
    M.fit(d.data, d.target)
    score = M.score(d.data, d.target)
    print(f"Score on {name}: {score}")
    return score

def out_of_sample_test(d, loss, name, alpha=.0001):
    M = SGDClassifier(loss=loss, n_jobs=-1, alpha=alpha)
    scores = cross_val_score(M, d.data, d.target)
    print(f"Score on {name}: {scores.mean()}, std: {scores.std()}")
    return scores.mean()


D1 = fetch_openml("one-hundred-plants-shape", as_frame=False, parser="auto")
D2 = fetch_openml("one-hundred-plants-margin", as_frame=False, parser="auto")
D3 = fetch_openml("one-hundred-plants-texture", as_frame=False, parser="auto")

# Testing the datasets for margin and texture
print("-------------------- testing concat on data --------------------")
d = concatenate((D2.data[:1599], D3.data), axis=1)
class container:
    def __init__(self, data, target):
        self.data = data
        self.target = target
D = container(d, D3.target)
test(D, "modified_huber", "concatenate")
out_of_sample_test(D, "modified_huber", "concatenate")

# making scalers for each dataset
scaler_shape = preprocessing.StandardScaler().fit(D1.data)
scaler_margin = preprocessing.StandardScaler().fit(D2.data)
scaler_texture = preprocessing.StandardScaler().fit(D3.data)

# scaling datasets
D1.data = scaler_shape.transform(D1.data)
D2.data = scaler_margin.transform(D2.data)
D3.data = scaler_texture.transform(D3.data)

# testing concatenate on scaled data
print("-------------------- testing concat on scaled data --------------------")
d = concatenate((D2.data[:1599], D3.data), axis=1)
D = container(d, D3.target)
test(D, "modified_huber", "concatenate")
out_of_sample_test(D, "modified_huber", "concatenate")

# Testing multiple loss functions
""" 
loss_functions = ["hinge", "log_loss", "modified_huber", "perceptron"]

sample = []

for loss in loss_functions:
    print(f"------------------- testing on {loss} function -------------------")
    in_sample = []
    in_sample.append(test(D1, loss, "shape"))
    in_sample.append(test(D2, loss, "margin"))
    in_sample.append(test(D3, loss, "texture"))

    print(f"Out of sample estimation starting")
    out_sample = []
    out_sample.append(out_of_sample_test(D1, loss, "shape"))
    out_sample.append(out_of_sample_test(D2, loss, "margin"))
    out_sample.append(out_of_sample_test(D3, loss, "texture"))
    
    sample.append((in_sample, out_sample))

print("------------------- testing on scaled data -------------------")

# making scalers for each dataset
scaler_shape = preprocessing.StandardScaler().fit(D1.data)
scaler_margin = preprocessing.StandardScaler().fit(D2.data)
scaler_texture = preprocessing.StandardScaler().fit(D3.data)

# scaling datasets
D1.data = scaler_shape.transform(D1.data)
D2.data = scaler_margin.transform(D2.data)
D3.data = scaler_texture.transform(D3.data)

sample_scaled = []

for loss in loss_functions:
    print(f"------------------- testing on {loss} function -------------------")
    in_sample = []
    in_sample.append(test(D1, loss, "shape"))
    in_sample.append(test(D2, loss, "margin"))
    in_sample.append(test(D3, loss, "texture"))

    print(f"Out of sample estimation starting")
    out_sample = []
    out_sample.append(out_of_sample_test(D1, loss, "shape"))
    out_sample.append(out_of_sample_test(D2, loss, "margin"))
    out_sample.append(out_of_sample_test(D3, loss, "texture"))

    sample_scaled.append((in_sample, out_sample))

with open("sgdout.csv", "w") as f:
    f.write("non-scaled data:\n")
    for loss in range(len(sample)):
        f.write(f"{loss_functions[loss]}\n")
        f.write(f",shape, margin, texture\n")
        f.write(f"in sample, {sample[loss][0][0]}, {sample[loss][0][1]},{sample[loss][0][2]}\n")
        f.write(f"out of sample, {sample[loss][1][0]}, {sample[loss][1][1]},{sample[loss][1][2]}\n")

    f.write("scaled data:\n")
    for loss in range(len(sample_scaled)):
        f.write(f"{loss_functions[loss]}\n")
        f.write(f",shape, margin, texture\n")
        f.write(f"in sample, {sample_scaled[loss][0][0]}, {sample_scaled[loss][0][1]},{sample_scaled[loss][0][2]}\n")
        f.write(f"out of sample, {sample_scaled[loss][1][0]}, {sample_scaled[loss][1][1]},{sample_scaled[loss][1][2]}\n")
""" 

























































































