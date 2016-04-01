def forward_multiply_gate(a, b)
  a * b
end

def forward_add_gate(a, b)
  a + b
end

def forward_circuit(x, y, z)
  q = forward_add_gate(x, y)
  f = forward_multiply_gate(q, z)
end

x = -2
y = 5
z = -4

q = forward_add_gate(x, y) # 3
f = forward_multiply_gate(q, z) # -12

# gradient of the MULTIPLY gate with respect to its inputs
# wrt is short for "with respect to"
derivative_f_wrt_z = q # 3
derivative_f_wrt_q = z # -4

# derivative of the ADD gate with respect to its inputs
derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

# chain rule
derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q # -4
derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q # -4

# final gradient, from above: [-4, -4, 3]
gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

# let the inputs respond to the force/tug:
step_size = 0.01;
x = x + step_size * derivative_f_wrt_x; # -2.04
y = y + step_size * derivative_f_wrt_y; # 4.96
z = z + step_size * derivative_f_wrt_z; # -3.97

# Our circuit now better give higher output:
q = forward_add_gate(x, y); # q becomes 2.92
f = forward_multiply_gate(q, z); # output is -11.59, up from -12! Nice!

# initial conditions
x = -2
y = 5
z = -4

# numerical gradient check
h = 0.0001
x_derivative = (forward_circuit(x+h,y,z) - forward_circuit(x,y,z)) / h # -4
y_derivative = (forward_circuit(x,y+h,z) - forward_circuit(x,y,z)) / h # -4
z_derivative = (forward_circuit(x,y,z+h) - forward_circuit(x,y,z)) / h # 3
