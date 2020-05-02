from typing import List, Union

# from all input neuron to one output
# weights[i] => to one output[i]
# (I)  \
# (I) --- => (O) => (input * weights[1] = output[1])
# (I)  /
weights = [
    [0.1, 0.1, -0.4],  # травм ?
    [0.1, 0.2, 0],  # побед ?
    [0, 1.3, 0.1]  # печаль ?
]


def w_sum(vectOne: List[float], vectTwo: List[float]) -> float:
  out = 0.0

  if len(vectOne) != len(vectTwo):
    print('Error ! w_sum, inputs size array are not equal')
    # return 0.0

  for i in range(len(vectOne)):
    out += vectOne[i] * vectTwo[i]

  return out


def vect_mat_mul(vect: List[float], matrix: List[List[float]]) -> List[float]:
  assert (len(vect) == len(matrix))
  output = [0, 0, 0]
  for i in range(len(vect)):
    output[i] = w_sum(vect, matrix[i])
  return output


def neural_network(input: List[float], weights: List[List[float]]) -> float:
  pred = vect_mat_mul(input, weights)

  return pred


# Input Data [toes[i], wlrec[i], nfans[i]]
toes = [8.5, 9.5, 9.9, 9]  # toes - текущее ср.число игр за сезон между игроками
wlrec = [0.65, 0.8, 0.8, 0.9]  # win / lose
nfans = [1.2, 1.3, 0.5, 1.0]  # number of fans (millions)

# Output Data [hurt[i], win[i], sad[i]]
hurt = [0.1, 0, 0, 0.1]  # hutred players
win = [1, 1, 0, 1]  # They will win ?
sad = [0.1, 0, 0.1, 0.2]  # Sadness =(

alpha = 0.01

input = [toes[0], wlrec[0], nfans[0]]
true = [hurt[0], win[0], sad[0]]
print('input', input)
print('true', true)

pred = neural_network(input, weights)
print('pred', pred)

error = [0, 0, 0]
delta = [0, 0, 0]

for i in range(len(true)):
  error[i] = (pred[i] - true[i])**2
  delta[i] = pred[i] - true[i]
print('errors', error)


# matrix with [X * Y] size matrix filled with zeroes
def zeros_matrix(sizeX, sizeY):
  matrix = [0] * sizeX
  for i in range(sizeX):
    matrix[i] = [0] * sizeY

  return matrix


def outer_pred(vec_a, vec_b):
  out = zeros_matrix(len(vec_a), len(vec_b))
  for i in range(len(vec_a)):
    for j in range(len(vec_b)):
      out[i][j] = vec_a[i] * vec_b[j]

  return out


weight_deltas = outer_pred(input, delta)
print('weight_deltas', weight_deltas)