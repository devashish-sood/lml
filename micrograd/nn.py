import random
from engine import * 

class Neuron: 

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1, 1), '') for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1), '')

  def __call__(self, x):
    act = sum((w * x for w, x in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def params(self): 
    return self.w + [self.b]
  
class Layer: 
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def params(self): 
    params = []
    for n in self.neurons: 
      params.extend(n.params())
    return params

  
class MLP: 
  def __init__(self, nin, nouts) -> None:
    self.layers = [Layer(i, o) for i, o in zip([nin] + nouts, nouts)]

  def __call__(self, x): 
    for l in self.layers: 
      x = l(x)
    
    return x
    
  def params(self): 
    params = []
    for l in self.layers: 
      params.extend(l.params())
    
    return params

      

    
if __name__ == "__main__": 
  # x = [2.0, 3.0]
  # p = MLP(2, [3, 4, 5, 1])
  # print(p(x))
  # n = Layer(2, 3)
  # print(n(x))

  # Training data
  xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
  ]
  ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

  n = MLP(3, [4, 4, 1])
  params = n.params()

  # print(n.layers[0].neurons[0].w[0].grad)
  # print(ypred)
  # print(len(n.params()))

  for _ in range(500):
    ypred = [n(x) for x in xs]
    loss = sum((yr - yp) ** 2 for yr, yp in zip(ys, ypred))
    print(loss)
    for p in params: 
      p.grad = 0.0
    loss.backward() 

    for p in params: 
      p.data += -0.05 * p.grad

  print(ypred)

