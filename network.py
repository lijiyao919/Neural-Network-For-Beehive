"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
# Third-party libraries
import numpy as np
import pylab as pl
import logging
import json



class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        logging.basicConfig(filename='NN.log', filemode='w', level=logging.DEBUG, format='%(message)s')
        self.logger = logging.getLogger("NN")
        self.log_tr = []
        self.log_te = []
        self.log_tr_bee = []
        self.log_te_bee = []
        self.log_tr_no_bee = []
        self.log_te_no_bee = []

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            training_bee_data=None,
            training_no_bee_data=None,
            test_bee_data=None,
            test_no_bee_data=None,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        epoch = []
        tr_data = []
        te_data = []
        tr_bee = []
        te_bee = []
        tr_no_bee = []
        te_no_bee = []

        if training_data: n_training_data = len(training_data)
        if test_data: n_test_data = len(test_data)
        if training_bee_data: n_training_bee_data = len(training_bee_data)
        if training_no_bee_data: n_training_no_bee_data = len(training_no_bee_data)
        if test_bee_data: n_test_bee_data = len(test_bee_data)
        if test_no_bee_data: n_test_no_bee_data = len(test_no_bee_data)
        print "           Train_data      Test_data       Tr_bee     Te_bee      Tr_no_bee      Te_no_bee"
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n_training_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            #log weights and biases
            #weights_log = [w.tolist() for w in self.weights]
            #biases_log = [b.tolist() for b in self.biases]
            #self.logger.info("Epoch %d : W: %s ", j, str(weights_log))
            #self.logger.info("Epoch %d : B: %s", j, str(biases_log))

            if test_data:
                print "Epoch {0}: {1} / {2} ; {3} / {4} ; {5} / {6} ; {7} / {8} ; {9} / {10} ; {11} / {12}".format(j, self.evaluate(training_data), n_training_data,
                                                                                                                      self.evaluate(test_data), n_test_data,
                                                                                                                      self.evaluate(training_bee_data), n_training_bee_data,
                                                                                                                      self.evaluate(test_bee_data),n_test_bee_data,
                                                                                                                      self.evaluate(training_no_bee_data), n_training_no_bee_data,
                                                                                                                      self.evaluate(test_no_bee_data), n_test_no_bee_data)
                epoch.append(j)
                tr_data.append(self.evaluate(training_data)/float(n_training_data))
                te_data.append(self.evaluate(test_data)/float(n_test_data))
                tr_bee.append(self.evaluate(training_bee_data)/float(n_training_bee_data))
                te_bee.append(self.evaluate(test_bee_data)/float(n_test_bee_data))
                tr_no_bee.append(self.evaluate(training_no_bee_data)/float(n_training_no_bee_data))
                te_no_bee.append(self.evaluate(test_no_bee_data)/float(n_test_no_bee_data))
            else:
                print "Epoch {0} complete".format(j)

            self.logger.info("Epoch %d", j)
            self.logger.info("tr: %s", str(self.log_tr))
            self.logger.info("te: %s", str(self.log_te))
            self.logger.info("tr_bee: %s", str(self.log_tr_bee))
            self.logger.info("te_bee: %s", str(self.log_te_bee))
            self.logger.info("tr_no_bee: %s", str(self.log_tr_no_bee))
            self.logger.info("te_no_bee: %s", str(self.log_te_no_bee))



        pl.subplot(311)
        tr_plot, = pl.plot(epoch, tr_data, 'b', label="tr")
        te_plot,  = pl.plot(epoch, te_data, 'r', label="te")
        pl.xlabel('Epoch')
        pl.ylabel('Correctness')
        pl.legend(handles=[tr_plot, te_plot], loc=1)

        pl.subplot(312)
        tr_bee_plot, = pl.plot(epoch, tr_bee, 'b', label="tr_bee")
        te_bee_plot, = pl.plot(epoch, te_bee, 'r', label="te_bee")
        pl.xlabel('Epoch')
        pl.ylabel('Correctness')
        pl.legend(handles=[tr_bee_plot, te_bee_plot], loc=1)

        pl.subplot(313)
        tr_no_bee_plot, = pl.plot(epoch, tr_no_bee, 'b', label="tr_no_bee")
        te_no_bee_plot, = pl.plot(epoch, te_no_bee, 'r', label="te_no_bee")
        pl.xlabel('Epoch')
        pl.ylabel('Correctness')
        pl.legend(handles=[tr_no_bee_plot, te_no_bee_plot], loc=1)

        pl.show()


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        #delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []

        for (x, y) in test_data:
            a = self.feedforward(x)

            if len(test_data) == 40806:
                self.log_tr.append((a, y))
            elif len(test_data) == 17439:
                self.log_te.append((a, y))
            elif len(test_data) == 2629:
                self.log_tr_bee.append((a, y))
            elif len(test_data) == 1076:
                self.log_te_bee.append((a, y))
            elif len(test_data) == 38177:
                self.log_tr_no_bee.append((a, y))
            elif len(test_data) == 16363:
                self.log_te_no_bee.append((a, y))
            else:
                self.logger.warning("No Such length dataset!")

            error = abs(a-y)
            if error > 0.5:
                test_results.append(False)
            else:
                test_results.append(True)

        return sum(int(x == True) for x in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"weights": [w.tolist() for w in self.weights], "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def restore(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network([1024, 30, 1])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
