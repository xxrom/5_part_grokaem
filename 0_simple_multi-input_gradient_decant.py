# Neural network
weights = [.1, .2, -.1]

# learn Info
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1]
# ans info
win_or_lose_binary = [1, 1, 0, 1]


# Sum two vectors
def w_sum(a, b):
  assert (len(a) == len(b))

  output = 0

  for i in range(len(a)):
    output += a[i] * b[i]

  return output


# ele_mul
def vector_multiplier(multiplierNumber, vector):
  output = [0, 0, 0]
  assert (len(output) == len(vector))

  for i in range(len(vector)):
    output[i] = multiplierNumber * vector[i]

  return output


def neural_network(input, weights):
  pred = w_sum(input, weights)
  return pred


# Input data
alpha = 0.1  # good choice = 0.01
# with alpha 0.1 - it will be discrapency(расхождение) in weight[0]
# it increases each iteration

true = win_or_lose_binary[0]
input = [toes[0], wlrec[0], nfans[0]]
print('INIT weights: ' + str(weights))
print('-------------')

for index in range(3):
  # Prediction and error calculation
  pred = neural_network(input, weights)
  error = (pred - true)**2
  delta = pred - true
  print('Iteration: %d' % index)
  print('Pred: %.04f' % pred)
  print('CubeError: %.04f' % error)
  print('Delta: %.04f' % delta)

  # Calc change delta (change weights)
  weight_deltas = vector_multiplier(delta, input)

  # Learning (main change in weights)
  for i in range(len(weights)):
    weights[i] -= alpha * weight_deltas[i]

  print('weights: ' + str(weights))
  print('weights deltas: ' + str(weight_deltas))

  print('-------------')
