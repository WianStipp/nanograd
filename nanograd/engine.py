"""
This module contains the autograd engine, from Kaparthy's micrograd.
"""

from typing import Tuple, List


class Value:
  def __init__(self, data, _children: Tuple['Value'] = (), _op: str = '', label: str = '') -> None:
    self.data = data
    self.grad = 0.0
    self._children = set(_children)
    self._op = _op
    self.label = label
    self._backward = lambda: None

  def __repr__(self) -> str:
    return f"Value(data={self.data}, grad={self.grad})"
    
  def __str__(self) -> str:
    return self.__repr__()
    
  def __add__(self, other) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    new = Value(self.data + other.data, (self, other), '+')
    def _backward():
      other.grad += 1.0 * new.grad
      self.grad += 1.0 * new.grad
    new._backward = _backward
    return new

  
  def __mul__(self, other) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    new = Value(self.data * other.data, (self, other), "*")
    def _backward():
      other.grad += self.data * new.grad
      self.grad += other.data * new.grad
    new._backward = _backward
    return new
  
  def backward(self) -> None:

    visited = set()
    topologically_sorted_nodes: List['Value'] = []
    def build_topologically_sorted_nodes(root: 'Value'):
      if root not in visited:
        visited.add(root)
        for child in root._children:
          build_topologically_sorted_nodes(child)
        topologically_sorted_nodes.append(root)

    build_topologically_sorted_nodes(self)

    print([v.label for v in topologically_sorted_nodes])
    self.grad = 1.0
    for node in reversed(topologically_sorted_nodes):
      node._backward()
  

if __name__ == "__main__":
  x = Value(0.0)
  y = Value(12.3)
  a = Value(1.0)
  b = Value(-4.0)
  c = a + b + x * y
  print(c.grad)
  print(a.grad, b.grad)
