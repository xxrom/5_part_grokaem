weights = [0.3, .2, .9]


# ele_mul
def vector_multiplier(number, weights):
  output = [0, 0, 0]
  assert (len(output) == len(weights))

  for i in range(len(weights)):
    output[i] = number * weights[i]

  return output


def neural_network(input, weights):
  pred = vector_multiplier(input, weights)
  return pred


wlrec = [.65, 1, 1, .9]

hurt = [.1, 0, 0, .1]
win = [1, 1, 0, 1]
sad = [.1, 0, .1, .2]

input = wlrec[0]
true = [hurt[0], win[0], sad[0]]

pred = neural_network(input, weights)

error = [0, 0, 0]
delta = [0, 0, 0]

for i in range(len(true)):
  error[i] = (pred[i] - true[i])**2
  delta[i] = pred[i] - true[i]

print(' error ', error)
print(' delta ', delta)


def scalar_vector_multiplier(inputVector, vector):
  output = [0] * len(vector)
  assert (len(inputVector) == len(vector))

  for i in range(len(vector)):
    output[i] = inputVector[i] * vector[i]

  return output


alpha = 0.1
# weight_deltas is equal to pred ??? ??? ??? I think so...
# TODO: is it corrent scalar_vector_multiplier ??? or should be vector_multiplier ???
weight_deltas = scalar_vector_multiplier(delta, weights)

# TODO: is it correct to calculate weights_delta again like that ?
for i in range(len(true)):
  weights[i] -= weight_deltas[i] * alpha

print(' new Weights', weights)
print(' weights_delta', weight_deltas)