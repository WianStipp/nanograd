from nanograd import plotting, engine

if __name__ == "__main__":
  x = engine.Value(0.0, label='x')
  y = engine.Value(12.3, label='y')
  z = x * y; z.label = 'z'
  a = engine.Value(1.0, label='a')
  b = engine.Value(-4.0, label='b')
  c = a + b; c.label = 'c'
  d = c + z; d.label = 'd'
  d.backward()
  plotting.draw_computation_graph(d)
