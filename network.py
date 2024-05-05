import sklearn.datasets
import sklearn.neural_network
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

# h1 = 1000, h2 = 1300, h3 = 900, h4 = 800, h5 = 600, h6 = 500, h7 = 350, h8 = 300, h9 = 150,

class NNClassifier(nn.Module):

    def __init__(self, input_features = 64 , h1 = 200, out_features = 100):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.out = nn.Linear(h1, out_features)


    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.out(x)

        return x

network = NNClassifier()

D = sklearn.datasets.fetch_openml('one-hundred-plants-texture', parser = 'liac-arff', as_frame = False)

X = D.data
y = D.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)

y_train = torch.from_numpy(np.array([ (int(num) - 1) for num in y_train ])).type(torch.LongTensor)
y_test = torch.from_numpy(np.array([ (int(num) - 1) for num in y_test ])).type(torch.LongTensor)


# loss function
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(network.parameters(), lr = 0.01)

epochs = 1000

for i in range(epochs):
    y_pred = network.forward(X_train)

    loss = criterion(y_pred, y_train)

    # print loss every 10 epochs
    if i % 10 == 0:
        print(f'epoch: {i}, loss: {loss}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
correct_predictions = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        pred = network.forward(data)

        if i == 0:
            print(pred)
            print(f'result: {pred.argmax().item()}')

        # prints {class} then {predicted class}
        # print(f'{i + 1}.) {y_test[i]} \t {pred.argmax().item()}')

        if pred.argmax().item() == y_test[i]:
            correct_predictions += 1

print(f'test sample correct prediction: {correct_predictions} (out of {len(y_test)} samples)')
print(f'out-sample error estimate: {1 - correct_predictions / len(y_test)}')

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_train):
        pred = network.forward(data)

        if pred.argmax().item() == y_train[i]:
            correct += 1

print(f'train sample correct predictions: {correct} (out of {len(y_train)} samples)')
print(f'in sample error: {1 - correct / len(y_train)}')