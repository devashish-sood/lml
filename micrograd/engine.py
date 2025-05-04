import math

class Value: 
  def __init__(self, data, label, _children = (), _op='') -> None:
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.grad = 0.0
    self._backward = lambda: None
    self.label = label
  
  def __repr__(self) -> str:
    return f"Value(data={self.data}, label={self.label})"
  
  def __add__(self, other): 
    out = Value(self.data + other.data, (self, other), '+')

    def _backward(): 
      self.grad = 1.0 * out.grad
      other.grad = 1.0 * out.grad
    out._backward = _backward
    return out 

  def __mul__(self, other): 
    out = Value(self.data * other.data, (self, other), '*')

    def _backward(): 
      self.grad = other.data * out.grad
      other.grad = self.data * out.grad
    out._backward = _backward
    return out 
  
  def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self, ), 'tanh')
    def _backward(): 
      self.grad = out.grad * (1 - t ** 2)
    out._backward = _backward
    return out

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

o.grad = 1
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()
print(n.grad, b.grad, x1w1x2w2.grad, x1w1.grad, x2w2.grad, w1.grad, w2.grad, x1.grad, x2.grad)