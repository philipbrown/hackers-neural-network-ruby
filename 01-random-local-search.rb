def forward_multiply_gate(x, y)
  x * y
end

x = -2
y = 3

tweak_amount = 0.01
best_out = -Float::INFINITY
best_x = x
best_y = y

100.times do
  x_try = x + tweak_amount * (rand(0.0..1.0) * 2 - 1)
  y_try = y + tweak_amount * (rand(0.0..1.0) * 2 - 1)
  out = forward_multiply_gate(x_try, y_try)

  if out > best_out
    best_out = out
    best_x = x_try
    best_y = y_try
  end
end

puts "best x #{best_x}\n"
puts "best y #{best_y}\n"
puts "best out #{best_out}\n"
