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
  
  def __rmul__(self, other) -> 'Value':
    return self * other
  
  def __pow__(self, exponent) -> 'Value':
    assert isinstance(exponent, (int, float)), f"only supports int or float, got {type(exponent)}" 
    new = Value(self.data ** exponent, (self, ), f'**{exponent}')
    def _backward():
      self.grad += exponent * (self.data ** (exponent - 1)) * new.grad
    new._backward = _backward
    return new
  
  def relu(self) -> 'Value':
    new = Value(max(self.data, 0.0), (self, ), 'ReLU')

    def _backward():
      self.grad += (self.data >= 0.0) * new.grad
    new._backward = _backward
    return new
  
  def __neg__(self) -> 'Value':
    new = Value(-self.data, (self, ), '-')
    def _backward():
      self.grad += -1.0 * new.grad
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

    self.grad = 1.0
    for node in reversed(topologically_sorted_nodes):
      node._backward()
