class Unit
  attr_accessor :value
  attr_accessor :grad

  def initialize(value, grad)
    @value = value
    @grad  = grad
  end
end

class MultiplyGate
  attr_accessor :u0
  attr_accessor :u1
  attr_accessor :utop

  def forward(u0, u1)
    # store pointers to input Units u0 and u1 and output unit utop
    @u0 = u0
    @u1 = u1

    @utop = Unit.new(u0.value * u1.value, 0.0)
  end

  def backward
    # take the gradient in output unit and chain it with the
    # local gradients, which we derived for multiply gate before
    # then write those gradients to those Units.
    @u0.grad += @u1.value * @utop.grad;
    @u1.grad += @u0.value * @utop.grad;
  end
end

class AddGate
  attr_accessor :u0
  attr_accessor :u1
  attr_accessor :utop

  def forward(u0, u1)
    @u0 = u0
    @u1 = u1
    @utop = Unit.new(u0.value + u1.value, 0.0)
  end

  def backward
    @u0.grad += 1 * @utop.grad;
    @u1.grad += 1 * @utop.grad;
  end
end

class SigmoidGate
  attr_accessor :u0
  attr_accessor :utop

  def forward(u0)
    @u0 = u0
    @utop = Unit.new(sigmoid(@u0.value), 0.0)
  end

  def backward
    s = sigmoid(@u0.value)
    @u0.grad += (s * (1 - s)) * @utop.grad
  end

  def sigmoid(x)
    1 / (1 + Math::E**-x)
  end
end

# create input units
a = Unit.new(1.0, 0.0)
b = Unit.new(2.0, 0.0)
c = Unit.new(-3.0, 0.0)
x = Unit.new(-1.0, 0.0)
y = Unit.new(3.0, 0.0)

# create the gates
mulg0 = MultiplyGate.new
mulg1 = MultiplyGate.new
addg0 = AddGate.new
addg1 = AddGate.new
sg0   = SigmoidGate.new

# do the forward pass
ax = mulg0.forward(a, x) # a*x = -1
by = mulg1.forward(b, y) # b*y = 6
axpby = addg0.forward(ax, by) # a*x + b*y = 5
axpbypc = addg1.forward(axpby, c) # a*x + b*y + c = 2
s = sg0.forward(axpbypc) # sig(a*x + b*y + c) = 0.8808

puts s.value

s.grad = 1.0
sg0.backward # writes gradient into axpbypc
addg1.backward # writes gradients into axpby and c
addg0.backward # writes gradients into ax and by
mulg1.backward # writes gradients into b and y
mulg0.backward # writes gradients into a and x

step_size = 0.01
a.value += step_size * a.grad # a.grad is -0.105
b.value += step_size * b.grad # b.grad is 0.315
c.value += step_size * c.grad # c.grad is 0.105
x.value += step_size * x.grad # x.grad is 0.105
y.value += step_size * y.grad # y.grad is 0.210

# do the forward pass
ax = mulg0.forward(a, x)
by = mulg1.forward(b, y)
axpby = addg0.forward(ax, by)
axpbypc = addg1.forward(axpby, c)
s = sg0.forward(axpbypc)

puts s.value

# check the numerical gradient
def forward_circuit_fast(a, b, c, x, y)
  1 / (1 + Math::E**-(a*x + b*y + c))
end

a = 1
b = 2
c = -3
x = -1
y = 3
h = 0.0001
puts a_grad = (forward_circuit_fast(a+h,b,c,x,y) - forward_circuit_fast(a,b,c,x,y))/h
puts b_grad = (forward_circuit_fast(a,b+h,c,x,y) - forward_circuit_fast(a,b,c,x,y))/h
puts c_grad = (forward_circuit_fast(a,b,c+h,x,y) - forward_circuit_fast(a,b,c,x,y))/h
puts x_grad = (forward_circuit_fast(a,b,c,x+h,y) - forward_circuit_fast(a,b,c,x,y))/h
puts y_grad = (forward_circuit_fast(a,b,c,x,y+h) - forward_circuit_fast(a,b,c,x,y))/h
