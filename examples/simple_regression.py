"""
This is a simple regression demo of nanograd.
"""
import math
import random
import matplotlib.pyplot as plt

from nanograd import nn, plotting

N_DATAPOINTS = 128
N_EPOCHS = 90
LEARNING_RATE = 0.01

def main():
  X = [[random.uniform(-1.0, 1.0)] for _ in range(N_DATAPOINTS)]
  Y = [x[0]**2 + (math.exp(x[0] * 0.8) if x[0] > 0.2 else 0.0) + random.normalvariate(0.0, 0.05) for x in X]

  X_TRAIN = X[:int(len(X) * 0.8)]
  X_TEST = X[int(len(X) * 0.8):]
  Y_TRAIN = Y[:int(len(Y) * 0.8)]
  Y_TEST = Y[int(len(Y) * 0.8):]

  model = nn.MLP([1, 4, 4, 1])
  print("number of params:", len(model.parameters()))
  pred = [model(x) for x in X]

  mse = lambda pred, label: (pred - label) ** 2

  for _ in range(N_EPOCHS):
    losses = []
    for x, y in zip(X_TRAIN, Y_TRAIN):

      for w in model.parameters():
        w.grad = 0.0

      pred = model(x)[0]
      loss = mse(pred, y)
      loss.backward()

      for w in model.parameters():
        w.data = w.data - LEARNING_RATE * w.grad

      losses.append(loss.data)
    print('mean loss:', sum(losses) / len(losses))

  preds = [model(x)[0].data for x in X_TEST]

  plt.figure(figsize=(12, 8))
  plt.title('y = x**2 + (exp(x * 0.8) if x >0.2 else 0.0)')
  plt.scatter(X_TRAIN, Y_TRAIN, alpha=0.2, label='train')
  plt.scatter(X_TEST, Y_TEST, label='label')
  plt.scatter(X_TEST, preds, label='prediction')
  plt.legend()
  plt.show()


if __name__ == "__main__":
  main()

