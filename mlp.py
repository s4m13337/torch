import numpy as np

def sigmoid(z):
    return 1. / (1. + np.exp(z))

def int_to_onehot(y, num_labels):
    ohe = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ohe[i, val] = 1
    return ohe

class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=43):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # Hidden layer
        # Input dimensions: [n_examples, n_features]
        # dot [n_hidden, n_features].T
        # Output dimensions: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # Input dimensions: [n_examples, n_hidden]
        # dot [n_classes, n_hidden].T
        # Output dimensions: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out) 

        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
    
        # Output layer weights

        # one-hot encoding
        y_ohe = int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape([0])
        d_a_out__d_y_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_y_out
        d_z_out__d_w_out = a_h

        d_loss__d_w_out = np.dot(delta_out.T, d_z_out__d_w_out)
        d_loss__d_b_out = np.sum(delta_out, axis=0)

        # Part 2: dLoss/dHiddenWeigths
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1. - a_h)
        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss_d_w_out, d_loss_d_b_out, d_loss_d_w_h, d_loss_d_b_h)