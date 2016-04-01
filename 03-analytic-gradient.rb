def forward_multiply_gate(x, y)
  x * y
end

x = -2
y = 3
out = forward_multiply_gate(x, y) # -6
x_gradient = y
y_gradient = x

step_size = 0.01
x += step_size * x_gradient # -2.03
y += step_size * y_gradient # 2.98
puts out_new = forward_multiply_gate(x, y) # 5.87
