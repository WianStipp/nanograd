import operator
from typing import List
import functools
import random

from nanograd import engine

class Neuron:
  def __init__(self, in_size: int, activation: str = 'relu') -> None:
    self.w = [engine.Value(random.normalvariate(0.0, 1.0)) for _ in range(in_size)]
    self.b = engine.Value(random.normalvariate(0.0, 1.0))
    self.activation = activation
  
  def __call__(self, x: List[engine.Value]) -> engine.Value:
    assert len(x) == len(self.w)
    out: engine.Value = sum(map(functools.partial(functools.reduce, operator.mul), zip(x, self.w)), self.b)
    return out

  def parameters(self):
    return self.w + [self.b]

class ReLU:
  def __call__(self, x: List[engine.Value]):
    return [x_.relu() for x_ in x]
  
  def parameters(self):
    return []

class Linear:
  def __init__(self, in_features: int, out_features: int) -> None:
    self.in_features = in_features
    self.out_features = out_features
    self.neurons: List[Neuron] = [Neuron(in_features) for _ in range(out_features)]

  def __call__(self, inputs: List[engine.Value]) -> List[engine.Value]:
    return [neuron(inputs) for neuron in self.neurons]
  
  def parameters(self):
    return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
  def __init__(self, layer_dims: List[int]) -> None:
    self.layer_dims = layer_dims
    self.layers = []
    for in_, out in zip(layer_dims, layer_dims[1:]):
      self.layers.append(Linear(in_, out))
      self.layers.append(ReLU())
    self.layers = self.layers[:-1]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [param for layer in self.layers for param in layer.parameters()]
