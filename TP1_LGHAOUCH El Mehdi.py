import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


# class Layer pour définir une couche
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # calcule la sortie Y d'une couche pour une entrée X donnée
    def forward_propagate(self, input):
        raise NotImplementedError

    # calcule dE/dX pour un dE/dY donné (et met à jour les paramètres)
    def back_propagate(self, output_errs, learning_rate):
        raise NotImplementedError


# class FCLayer pour définir la couche fully-connected
class fullyConnectedLayer(Layer):
    # input_size = nombre de neurones d'entrée
    # output_size = nombre de neurones de sortie
    def __init__(self, input_size, output_size):
        self.w = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(1, output_size) - 0.5

    # renvoie la sortie pour une entrée donnée
    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.w) + self.b
        return self.output

    # calcule dE/dW, dE/dB pour une output_err=dE/dY donnée.
    # Renvoie input_err=dE/dX.
    def back_propagate(self, output_err, learning_rate):
        input_err = np.dot(output_err, self.w.T)
        weights_err = np.dot(self.input.T, output_err)

        # mise à jour des paramétres
        self.w -= learning_rate * weights_err
        self.b -= learning_rate * output_err
        return input_err


class ActivationLayer(Layer):
    def __init__(self, activation, activation_derive):
        self.activation = activation
        self.activation_derive = activation_derive

    # renvoie l'entrée activée
    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Renvoie input_err=dE/dX pour un output_err=dE/dY donné.
    # learning_rate n'est pas utilisé car il n'y a pas de paramètres « apprenables ».
    def back_propagate(self, output_err, learning_rate):
        return self.activation_derive(self.input) * output_err


# class du notre Network
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derive = None

    # ajouter 'layer' au 'network'
    def add(self, layer):
        self.layers.append(layer)

    # définir 'loss' à utiliser
    def use(self, loss, loss_derive):
        self.loss = loss
        self.loss_derive = loss_derive

    # prédire la sortie pour une entrée donnée
    def predict(self, input_data):
        # première dimension de l'échantillon
        samples = len(input_data)
        result = []

        # exécuter 'network' sur tous les échantillons
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagate(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # première dimension de l'échantillon
        echant = len(x_train)
        cout_ = []
        my_err = []
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(echant):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagate(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_derive(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.back_propagate(error, learning_rate)
                my_err.append(error)
            # calculate average error on all samples
            cout_.append(err)
            err /= echant

            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
        return cout_, my_err

    pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derive(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derive(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


if __name__ == "__main__":
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    net = Network()
    net.add(fullyConnectedLayer(2, 3))
    net.add(ActivationLayer(sigmoid, sigmoid_derive))
    net.add(fullyConnectedLayer(3, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_derive))

    # train
    net.use(mse, mse_derive)
    cost_, my_err = net.fit(x_train, y_train, epochs=7500, learning_rate=0.2)

    # test
    out = net.predict(x_train)
    print(out)

    import matplotlib.pyplot as plt

    plt.plot(cost_)
