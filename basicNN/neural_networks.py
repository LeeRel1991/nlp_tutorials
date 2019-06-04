"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: neural_networks.py
@time: 2019/6/4 14:59
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def generate_random_datasets(display=True):
    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)

    if display:
        plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
        plt.show()

    return X, y


def logistic_regression(X, y, display=True):
    # Train the logistic rgeression classifier
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)

    # Plot the decision boundary
    plot_decision_boundary(X, y, lambda x: clf.predict(x))
    plt.title("Logistic Regression")
    if display:
        plt.show()


def nn_regression(data, label, display=True):
    num_examples = len(data)  # training set size
    input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

    def build_model(hiden_dim, num_iter=20000, print_loss=False):
        """
        This function learns parameters for the neural network and returns the model_params.
        Args:
            hiden_dim: Number of nodes in the hidden layer
            num_iter: Number of iteration through the training data for gradient descent
            print_loss: If True, print the loss every 1000 iterations

        Returns:

        """
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        weight_xh = np.random.randn(input_dim, hiden_dim) / np.sqrt(input_dim)
        bias_xh = np.zeros((1, hiden_dim))
        weigh_ho = np.random.randn(hiden_dim, nn_output_dim) / np.sqrt(hiden_dim)
        bias_ho = np.zeros((1, nn_output_dim))

        # This is what we return at the end
        model_params = {}

        # Gradient descent. For each batch...
        for i in range(0, num_iter):
            # Forward propagation
            z_hiden = data.dot(weight_xh) + bias_xh
            a_hiden = np.tanh(z_hiden)
            z_out = a_hiden.dot(weigh_ho) + bias_ho
            exp_scores = np.exp(z_out)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(num_examples), label] -= 1
            dW2 = a_hiden.T.dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(weigh_ho.T) * (1 - np.power(a_hiden, 2))
            dW1 = np.dot(data.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (bias_xh and bias_ho don't have regularization terms)
            dW2 += reg_lambda * weigh_ho
            dW1 += reg_lambda * weight_xh

            # Gradient descent parameter update
            weight_xh += -epsilon * dW1
            bias_xh += -epsilon * db1
            weigh_ho += -epsilon * dW2
            bias_ho += -epsilon * db2

            # Assign new parameters to the model_params
            model_params = {'weight_xh': weight_xh, 'bias_xh': bias_xh, 'weigh_ho': weigh_ho, 'bias_ho': bias_ho}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, calculate_loss(model_params)))

        return model_params

    def calculate_loss(model):
        """
        Helper function to evaluate the total loss on the dataset
        Args:
            model_params:

        Returns:

        """
        W1, b1, W2, b2 = model['weight_xh'], model['bias_xh'], model['weigh_ho'], model['bias_ho']
        # Forward propagation to calculate our predictions
        z1 = data.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), label])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / num_examples * data_loss

    def predict(model, x):
        """
        Helper function to predict an output (0 or 1)
        Args:
            model_params:
            x:

        Returns:

        """
        W1, b1, W2, b2 = model['weight_xh'], model['bias_xh'], model['weigh_ho'], model['bias_ho']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    # Build a model_params with a 3-dimensional hidden layer
    model = build_model(3, print_loss=True)
    # Plot the decision boundary
    plot_decision_boundary(data, label, lambda x: predict(model, x))
    plt.title("Decision Boundary for hidden layer size 3")
    
    # 改变hiden layer 的神经元个数
    plt.figure(figsize=(16, 32))
    hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer size %d' % nn_hdim)
        model = build_model(nn_hdim)
        plot_decision_boundary(data, label, lambda x: predict(model, x))
    plt.show()
    

def main():
    data, label = generate_random_datasets(display=False)
    print("data: ", data.shape)
    # logistic_regression(data, label, display=True)
    nn_regression(data, label, display=True)


if __name__ == "__main__":
    main()