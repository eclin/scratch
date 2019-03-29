import numpy as np
import activations as act

class NeuralNetwork:
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    self.training_data = None
    self.test_data = None
    self.epochs = 1
    self.batch_size = 32
    self.learning_rate = 0.1

  def predict(self, a):
    """Return the output of the network if "a" is input."""
    for i in range(self.num_layers - 1):
      dot = np.dot(self.weights[i], a)
      print("a: ", a)
      print("w: ", self.weights[i])
      print("b: ", self.biases[i])
      print("dot: ", dot)
      a = act.sigmoid(dot + self.biases[i])
    return a

  def train(self):
    if not self.training_data:
      print "No training data!"
      return
    
    for i in range(self.epochs):
      random.shuffle(training_data)
      batches = np.split(training_data, batch_size)

      for batch in batches:
        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(w.shape) for w in self.weights]
        # Run gradient descent for each batch in batches

        for x, y in batch:
          delta_b, delta_w = backprop(x, y)
          change_b = [cb + db for cb, db in zip(change_b, delta_b)]
          change_w = [cw + dw for cw, dw in zip(change_w, delta_w)]

        self.weights = [w - (self.learning_rate / len(batch)) * nw 
                        for w, nw in zip(self.weights, change_w)]
        self.biases = [b - (self.learning_rate / len(batch)) * nb 
                       for b, nb in zip(self.biases, change_b)]

  def backprop(self, x, y):
    # initialize empty arrays to represent the change in the weights and biases
    delta_b = [np.zeros(b.shape) for b in self.biases]
    delta_w = [np.zeros(w.shape) for w in self.weights]

    # Feedforward
    a = x
    activations = [x]
    products = []
    for i in range(self.num_layers - 1):
      # z^l = w^l a^l-1 + b^l
      z = np.dot(self.weights[i], a) + self.biases
      products.append(z)
      a = act.sigmoid(z)
      activations.append(a)

    # Output Error
    delta = (activations[-1] - y) * act.sigmoid_derivative(products[-1])
    delta_b[-1] = delta
    delta_w[-1] = np.dot(delta, activations[-2].transpose())

    # Backprop Error
    errors = []
    for l in range(self.num_layers - 2, 0):
      delta = np.dot(self.weights[l + 1].transpose(), delta) * act.sigmoid_derivative(products[l])
      delta_b[l] = delta
      delta_w[l] = np.dot(delta, activations[l - 1])

    return (delta_b, delta_w)



if __name__ == '__main__':
  network = NeuralNetwork([3, 4, 1])
  print(network.num_layers)
  print(network.sizes)
  print(network.biases)
  print(network.weights)

  print("My predict:")
  print(network.predict(np.array([[0], [0], [1]])))
  
  network.train()
