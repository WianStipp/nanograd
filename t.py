from nanograd import plotting, engine, nn

if __name__ == "__main__":
  x = engine.Value(0.0, label='x')
  y = engine.Value(12.3, label='y')
  z = x * y; z.label = 'z'
  a = engine.Value(1.0, label='a')
  b = engine.Value(-4.0, label='b')
  c = a + b; c.label = 'c'
  d = c**2 + z; d.label = 'd'
  model = nn.MLP([3, 10, 10, 1])
  out = sum(model([d, c, b]))
  out.relu()
  out.backward()
  plotting.draw_computation_graph(out)
