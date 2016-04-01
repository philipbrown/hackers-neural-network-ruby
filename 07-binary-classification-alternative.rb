data = []; labels = []
data.push([1.2, 0.7]); labels.push(1)
data.push([-0.3, -0.5]); labels.push(-1)
data.push([3.0, 0.1]); labels.push(1)
data.push([-0.1, -1.0]); labels.push(-1)
data.push([-1.0, 1.1]); labels.push(-1)
data.push([2.1, -3]); labels.push(1)

a = 1; b = -2; c = -1

# Training
400.times do |iter|
  # Pick a random data point
  i = (rand * data.length).floor
  x = data[i][0]
  y = data[i][1]
  label = labels[i]

  # compute pull
  score = a*x + b*y + c
  pull = 0.0
  if (label === 1 && score < 1)
    pull = 1
  end
  if (label === -1 && score > -1)
    pull = -1
  end

  # computer gradient and update parameters
  step_size = 0.01
  a += step_size * (x * pull - a) # -a is from the regularization
  b += step_size * (y * pull - b) # -b is from the regularization
  c += step_size * (1 * pull)

  if (iter % 25 == 0) # every 10 iterations
    num_correct = 0
    i = 0
    data.each do |data|
      x = data[0]
      y = data[1]
      true_label = labels[i]

      predicted_label = a*x + b*y + c > 0 ? 1 : -1

      if predicted_label === true_label
        num_correct = num_correct + 1
      end
      i = i + 1
    end

    puts num_correct.to_f / data.length
  end
end
