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
    @u0 = u0
    @u1 = u1

    @utop = Unit.new(u0.value * u1.value, 0.0)
  end

  def backward
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

# A circuit: it takes 5 Units (x,y,a,b,c) and outputs a single Unit
# It can also compute the gradient w.r.t. its inputs

class Circuit
  attr_accessor :mulg0
  attr_accessor :mulg1
  attr_accessor :addg0
  attr_accessor :addg1

  attr_accessor :ax
  attr_accessor :by
  attr_accessor :axpby
  attr_accessor :axpbypc

  def initialize
    @mulg0 = MultiplyGate.new
    @mulg1 = MultiplyGate.new
    @addg0 = AddGate.new
    @addg1 = AddGate.new
  end

  def forward(x, y, a, b, c)
    @ax = @mulg0.forward(a, x) # a*x
    @by = @mulg1.forward(b, y) # b*y
    @axpby = @addg0.forward(@ax, @by) # a*x + b*y
    @axpbypc = @addg1.forward(@axpby, c) # a*x + b*y + c
  end

  def backward(gradient_top)
    @axpbypc.grad = gradient_top
    @addg1.backward # sets gradient in axpby and c
    @addg0.backward # sets gradient in ax and by
    @mulg1.backward # sets gradient in b and y
    @mulg0.backward # sets gradient in a and x
  end
end

# SVM
class SVM
  attr_accessor :a
  attr_accessor :b
  attr_accessor :c
  attr_accessor :circuit
  attr_accessor :unit_out

  def initialize
    @a = Unit.new(rand(-3..3).to_f, 0.0)
    @b = Unit.new(rand(-3..3).to_f, 0.0)
    @c = Unit.new(rand(-3..3).to_f, 0.0)

    @circuit = Circuit.new
  end

  def forward(x, y) # assume x and y are Units
    @unit_out = @circuit.forward(x, y, @a, @b, @c)
  end

  def backward(label) # label is +1 or -1
    # reset pulls on a, b, c
    @a.grad = 0.0
    @b.grad = 0.0
    @c.grad = 0.0

    # compute the pull based on what the circuit output was
    pull = 0.0

    if (label === 1 and @unit_out.value < 1)
      pull = 1 # the score was too low: pull up
    end

    if (label === -1 and @unit_out.value > -1)
      pull = -1 # the score was too high for a positive example, pull down
    end

    @circuit.backward(pull) # writes gradient into x,y,a,b,c

    # add regularization pull for parameters: towards zero and proportional to value
    @a.grad += -@a.value;
    @b.grad += -@b.value;
  end

  def learn_from(x, y, label)
    forward(x, y) # forward pass (set .value in all Units)
    backward(label) # backward pass (set .grad in all Units)
    parameter_update
  end

  def parameter_update
    step_size = 0.01
    @a.value += step_size * @a.grad
    @b.value += step_size * @b.grad
    @c.value += step_size * @c.grad
  end
end

data = []; labels = []
data.push([1.2, 0.7]); labels.push(1)
data.push([-0.3, -0.5]); labels.push(-1)
data.push([3.0, 0.1]); labels.push(1)
data.push([-0.1, -1.0]); labels.push(-1)
data.push([-1.0, 1.1]); labels.push(-1)
data.push([2.1, -3]); labels.push(1)
svm = SVM.new

# a function that computes the classification accuracy
eval_training_accuracy = ->(svm, data, labels) do
  num_correct = 0
  i = 0
  data.each do |item|
    x = Unit.new(item[0], 0.0)
    y = Unit.new(item[1], 0.0)
    true_label = labels[i]

    predicted_label = svm.forward(x, y).value > 0 ? 1 : -1

    if predicted_label === true_label
      num_correct = num_correct + 1
    end
    i = i + 1
  end

  num_correct.to_f / data.length
end

1000.times do |iter|
  # pick a random data point
  i = (rand * data.length).floor
  x = Unit.new(data[i][0], 0.0)
  y = Unit.new(data[i][1], 0.0)
  label = labels[i]
  svm.learn_from(x, y, label)

  if (iter % 25 == 0) # every 10 iterations
    accuracy = eval_training_accuracy.call(svm, data, labels)
    puts "training accuracy at iter #{iter}: #{accuracy}"

    if accuracy == 1.0
      puts "training finished"
      break
    end
  end
end
