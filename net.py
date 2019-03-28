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


  def backprop(self, x, y):



if __name__ == '__main__':
  network = NeuralNetwork([3, 4, 1])
  print(network.num_layers)
  print(network.sizes)
  print(network.biases)
  print(network.weights)

  print("My predict:")
  print(network.predict(np.array([[0], [0], [1]])))
  
  network.train()
