"""
This module contains code to plot various things.
"""

from typing import Set, Tuple
import graphviz

from nanograd import engine

def draw_computation_graph(value: engine.Value) -> None:
  dot = graphviz.Digraph('computation graph', graph_attr={"rankdir": "LR"})
  nodes, edges = trace(value)
  
  for node in nodes:
    uid = str(id(node))
    dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (node.label, node.data, node.grad), shape='record')
    if op := node._op:
      dot.node(name=uid+op, label=op)
      dot.edge(uid + op, uid)
  
  for from_node, to_node in edges:
    dot.edge(str(id(from_node)), str(id(to_node)) + to_node._op)

  dot.render(view=True)

def trace(root_value: engine.Value) -> Tuple[Set[engine.Value], Set[engine.Value]]:
  nodes, edges = set(), set()
  def build(v: engine.Value):
    if v not in nodes:
      nodes.add(v)
      for child in v._children:
        edges.add((child, v))
        build(child)
  build(root_value) 
  return nodes, edges
