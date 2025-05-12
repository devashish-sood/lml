import math

class Value: 
  def __init__(self, data, label, _children = (), _op='') -> None:
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.grad = 0.0
    self.label = label
    self._backward = lambda: None
  
  def __repr__(self) -> str:
    return f"Value(data={self.data})"
  
  def __add__(self, other): 
    other = other if isinstance(other, Value) else Value(other, 'cast_' + str(other))
    out = Value(self.data + other.data, 'add_' + self.label + '_' + other.label, (self, other), '+')
    def _backward(): 
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out 
  
  def __radd__(self, other): 
    return self + other
  
  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)
  
  def __rsub__(self, other): 
    return self - other

  def __mul__(self, other): 
    other = other if isinstance(other, Value) else Value(other, 'cast_' + str(other))
    out = Value(self.data * other.data, 'mul_' + self.label + '_' + other.label, (self, other), '*')
    def _backward(): 
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out 
  
  def __rmul__(self, other): 
    return self * other
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), 'exp_' + self.label, (self, ), 'exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out
  
  def __pow__(self, other): 
    x = self.data 
    assert(isinstance(other, (int, float)))
    out = Value(math.pow(x, other), 'pow_' + self.label, (self, ), 'pow')

    def _backward(): 
      self.grad += (other * (math.pow(x, other - 1))) * out.grad

    out._backward = _backward
    return out

  def __truediv__(self, other): 
    return self * (other ** - 1)
  
  def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, 'tanh_' + self.label, (self, ), 'tanh')
    def _backward(): 
      self.grad += out.grad * (1 - t ** 2)
    out._backward = _backward
    return out
  
  def backward(self): 
    topo = []
    visited = set()

    def t_sort(v): 
      if v not in visited: 
        visited.add(v)
        for child in v._prev: 
          t_sort(child)
        topo.append(v)
    
    t_sort(self)
    self.grad = 1.0
    list(map(lambda n: n._backward(), reversed(topo)))


if __name__ == "__main__": 

  # inputs x1,x2
  x1 = Value(2.0, label='x1')
  x2 = Value(0.0, label='x2')
  # weights w1,w2
  w1 = Value(-3.0, label='w1')
  w2 = Value(1.0, label='w2')
  # bias of the neuron
  b = Value(6.8813735870195432, label='b')
  x1*w1 + x2*w2 + b
  x1w1 = x1*w1; x1w1.label = 'x1*w1'
  x2w2 = x2*w2; x2w2.label = 'x2*w2'
  x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
  n = x1w1x2w2 + b; n.label = 'n'
  o = ((n * 2).exp() - 1) / ((n * 2).exp() + 1)

  # o.grad = 1
  # o._backward()
  # n._backward()
  # b._backward()
  # x1w1x2w2._backward()
  # x2w2._backward()
  # x1w1._backward()

  o.backward()


  print(f'grad\no: {o.grad}, n: {n.grad}, b: {b.grad}, x1w1x2w2: {x1w1x2w2.grad}, x1w1: {x1w1.grad}, x2w2: {x2w2.grad}, w1: {w1.grad}, w2: {w2.grad}, x1: {x1.grad}, x2: {x2.grad}')

  print(f'data\no: {o.data}, n: {n.data}, x1w1x2w2: {x1w1x2w2.data}, x2w2: {x2w2.data}, x1w1: {x1w1.data}, b: {b.data}, w2: {w2.data}, w1: {w1.data}, x2: {x2.data}, x1: {x1.data}')



  # print(x1 * 2)
  # print(2 * x1)
  # print(x1 + 5)
  # print(5 + x1)

  # a = Value(2.0, label='a')
  # b = Value(4.0, label='b')
  # print(a - b)