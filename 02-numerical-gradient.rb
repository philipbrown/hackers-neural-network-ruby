def forward_multiply_gate(x, y)
  x * y
end

x = -2
y = 3
out = forward_multiply_gate(x, y) # -6
h = 0.0001

# compute derivative with respect to x
xph = x + h # -1.9999
out2 = forward_multiply_gate(xph, y) # -5.9997
x_derivative = (out2 - out) / h # 3.0

# compute derivative with respect to y
yph = y + h # 3.0001
out3 = forward_multiply_gate(x, yph) # -6.0002
y_derivative = (out3 - out) / h # -2.0

step_size = 0.01
x = x + step_size * x_derivative # x becomes -1.97
y = y + step_size * y_derivative # y becomes 2.98
puts out_new = forward_multiply_gate(x, y) # -5.87
