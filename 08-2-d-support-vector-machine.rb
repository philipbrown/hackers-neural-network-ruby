x = [ [1.2, 0.7], [-0.3, 0.5], [3, 2.5] ] # array of 2-dimensional data
y = [1, -1, 1] # array of labels
w = [0.1, 0.2, 0.3] # example: random numbers
alpha = 0.1; # regularization strength

def cost(x, y, w, alpha)
  total_cost = 0.0; # L, in SVM loss function above
  n = x.length

  n.times do |i|
    # loop over all data points and compute their score
    xi = x[i]
    score = w[0] * xi[0] + w[1] * xi[1] + w[2]

    # accumulate cost based on how compatible the score is with the label
    yi = y[i] # label
    costi = [0, - yi * score + 1].max
    puts "example #{i}: xi = (#{xi}) and label = #{yi}"
    puts " score computed to be #{score}"
    puts " => cost computed to be #{costi}"
    total_cost += costi
  end

  # regularization cost: we want small weights
  reg_cost = alpha * (w[0]*w[0] + w[1]*w[1])
  puts "regularization cost for current model is #{reg_cost}"
  total_cost += reg_cost

  puts "total cost is #{total_cost}"

  total_cost
end

cost(x,y,w,alpha)
