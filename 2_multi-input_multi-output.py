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

alpha = 0.009

# input = [toes[0], wlrec[0], nfans[0]]
# trueArray = [hurt[0], win[0], sad[0]]
# print('input', input)
# print('true', true)


# learning function
def learnIteration(inputIterationNumber: int, isShowPrints=False):
  input = [toes[inputIteration], wlrec[inputIteration], nfans[inputIteration]]
  trueArray = [hurt[inputIteration], win[inputIteration], sad[inputIteration]]

  pred = neural_network(input, weights)
  # print('pred', pred)

  error = [0, 0, 0]
  delta = [0, 0, 0]

  for i in range(len(trueArray)):
    # Cube error, firstly solve big problems,
    # and only after it, solve small problems
    error[i] = (pred[i] - trueArray[i])**2
    #
    delta[i] = pred[i] - trueArray[i]
    if isShowPrints:
      print('errors', error)
    weight_deltas = outer_pred(input, delta)
    # print('weight_deltas', weight_deltas)

    for j in range(len(weights[i])):
      # Apply changes  for each output weight
      # maybe it should be applied in other way ? =D
      weights[i][j] -= weight_deltas[j][i] * alpha


for ii in range(300):
  # I'm not sure that it's correct way to teach NN
  # Maybe it is better calculate AVG error for each input
  # and apply it ...
  # Change weights after each input
  for inputIteration in range(len(toes)):
    learnIteration(inputIteration)

learnIteration(0, True)
'''
alpha = 0.001
ii = 25000

errors [0.0011077411992017857, 0, 0]
errors [0.0011077411992017857, 0.01770334244762078, 0]
errors [0.0011077411992017857, 0.01770334244762078, 0.00035452208105551984]
errors [0.002495604692682505, 0, 0]
errors [0.002495604692682505, 0.000422134330389761, 0]
errors [0.002495604692682505, 0.000422134330389761, 0.0099318711176652]
errors [0.0047715867202401206, 0, 0]
errors [0.0047715867202401206, 0.011624229312993135, 0]
errors [0.0047715867202401206, 0.011624229312993135, 0.006578673673701122]
errors [4.7293735817031664e-05, 0, 0]
errors [4.7293735817031664e-05, 0.001245905980574472, 0]
errors [4.7293735817031664e-05, 0.001245905980574472, 8.956777941957558e-05]
errors [0.001107726101314892, 0, 0]
errors [0.001107726101314892, 0.01770266701784014, 0]
errors [0.001107726101314892, 0.01770266701784014, 0.0003544733836479419]

-----------------
alpha = 0.009
ii = 300

learnIteration(0, True)
errors [0.0008073293403828909, 0, 0]
errors [0.0008073293403828909, 0.01051218484162111, 0]
errors [0.0008073293403828909, 0.01051218484162111, 0.00020163150186320943]
'''