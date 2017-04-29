from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils


def prepare_dataset_for_NN(dataset, img_shape, img_channels): 
    if len(dataset) == 2:
        X = dataset[0]
        y = dataset[1]
        assert X.ndim == 2, "Do not need to reshape!"
        img_rows = img_shape[0]
        img_cols = img_shape[1]
        X = X.reshape(-1, img_rows, img_cols, img_channels, order = 'F')
        X = X.transpose((0,2,1,3))
        X = X.astype('float32')
        X /= np.max(X)
        if y is not None:
            y = np_utils.to_categorical(y.astype('int'), len(np.unique(y)))
        return (X, y)
    elif len(dataset) == 3:
        X = dataset[0]
        y = dataset[1]
        sample_weight = dataset[2]
        assert X.ndim == 2, "Do not need to reshape!"
        img_rows = img_shape[0]
        img_cols = img_shape[1]
        X = X.reshape(-1, img_rows, img_cols, img_channels, order = 'F')
        X = X.transpose((0,2,1,3))
        X = X.astype('float32')
        X /= np.max(X)
        if y is not None:
            y = np_utils.to_categorical(y.astype('int'), len(np.unique(y)))
        return (X, y, sample_weight)


def cons_CNN_MNIST(num_category = 2, nb_filters = 32, kernel_size = (3,3), input_shape = (28, 28, 1), pool_size = (2,2), activation = "relu"):
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape = input_shape))
    model.add(Activation(activation))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))

    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(num_category))
    model.add(Activation('softmax'))

    weights = model.get_weights()

    return model, weights


def cons_CNN_CIFAR(num_category = 2, nb_filters = 32, kernel_size = (3,3), input_shape = (32, 32, 3), pool_size = (2,2), activation = "relu"):
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', input_shape = input_shape))
    model.add(Activation(activation))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same'))
    model.add(Activation(activation))
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    model.add(Dense(num_category))
    model.add(Activation('softmax'))

    weights = model.get_weights()

    return model, weights

def shuffle_weights(model, weights=None):
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


class CNN(object):
    def __init__(
        self, 
        dataset_name = None, 
        num_category = 2, 
        img_shape = (28, 28), 
        img_channels = 1, 
        epochs = 10, 
        early_stopping = False, 
        early_stopping_validation_fraction = 0.1, 
        early_stopping_patience = 10,
        validation_data_for_early_stopping = None,
        verbose = 1, 
    ):
        """ The CNN classifier class works in the same way as sklearn's LogisticRegression
        class. The __init__() function takes in the following parameters:

        Parameters
        ----------
          dataset_name : str
            Choose between "mnist" or "cifar". Othervise leave it as None.

          num_category : int
            Number of output categories. Default is 2 for binary classification

          img_shape : 2-element tuple
            The height and width of image shape for input. If dataset_name == None, 
            the user should specify the image shape.

          img_channels : int
            Number of color channels for the image. Default is 1.

          epochs : int
            Maximum training epochs for the neural network

          early_stopping : bool
            Set to True to allow for early stopping. Default is False

          early_stopping_validation_fraction : float
            Fraction of training examples used for the validation set for early stopping.
            Only effective when early_stopping is True and validation_data_for_early_stopping is None.

          early_stopping_patience : int
            The number of epochs with no improvement after which training will be stopped.

          validation_data_for_early_stopping : tuple 
            Validation data, in the form of (validation_examples, validation_label) 
            or (validation_examples, validation_label, validation_sample_weight). Default is None

          verbose : int
            0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        """
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.img_shape = img_shape
        self.img_channels = img_channels
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.early_stopping_validation_fraction = early_stopping_validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.validation_data_for_early_stopping = validation_data_for_early_stopping
        if self.dataset_name == "mnist":
            self.clf, self.weights = cons_CNN_MNIST(num_category = num_category)
            self.img_shape = (28, 28)
            self.img_channels = 1
            self.batch_size = 128
            self.optimizer = 'adadelta'
        elif self.dataset_name == "cifar":
            self.clf, self.weights = cons_CNN_CIFAR(num_category = num_category)
            self.img_shape = (32, 32)
            self.img_channels = 3
            self.batch_size = 32
            self.optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            self.clf, self.weights = cons_CNN_MNIST(num_category = num_category, input_shape = (self.img_shape[0], self.img_shape[1], self.img_channels))
            self.batch_size = 32
            self.optimizer = 'adadelta'

    def fit(self, X, s, epochs = None, sample_weight = None, verbose = None):
        shuffle_weights(self.clf, self.weights)
        self.clf.compile(loss = 'categorical_crossentropy', optimizer = self.optimizer, metrics=['accuracy'])
        if epochs is not None:
            self.epochs = epochs
        if verbose is None:
            verbose = self.verbose

        if self.early_stopping:
            callbacks = [EarlyStopping(monitor = 'val_loss', patience = self.early_stopping_patience, verbose = 1)]
            if self.validation_data_for_early_stopping is None:
                if sample_weight is None:
                    X_train, X_val, s_train, s_val = train_test_split(X, s, test_size = self.early_stopping_validation_fraction, stratify = s)
                    sample_weight_train = None
                    validation_data = (X_val, s_val)
                else:
                    X_train, X_val, s_train, s_val, sample_weight_train, sample_weight_val = train_test_split(X, s, sample_weight, test_size = self.early_stopping_validation_fraction, stratify = s)
                    validation_data = (X_val, s_val, sample_weight_val)
            else:
                X_train = X
                s_train = s
                sample_weight_train = sample_weight
                validation_data = self.validation_data_for_early_stopping
            validation_data_for_NN = prepare_dataset_for_NN(validation_data, self.img_shape, self.img_channels)
        else:
            callbacks = None
            X_train = X
            s_train = s
            sample_weight_train = sample_weight
            if self.validation_data_for_early_stopping is not None:
                validation_data_for_NN = prepare_dataset_for_NN(self.validation_data_for_early_stopping, self.img_shape, self.img_channels)
            else:
                validation_data_for_NN = None

        X_train, s_train = prepare_dataset_for_NN((X_train, s_train), self.img_shape, self.img_channels)
            
        self.clf.fit(X_train, s_train, sample_weight = sample_weight_train, batch_size = self.batch_size, nb_epoch = self.epochs, validation_data = validation_data_for_NN, callbacks = callbacks, shuffle = True, verbose = verbose)
        
        return self

    def save_model(self, filename):
        self.clf.save(filename)

    def load_model(self, filename):
        self.clf = load_model(filename)

    def predict(self, X):
        (X, _) = prepare_dataset_for_NN((X, None), self.img_shape, self.img_channels)
        preds = (self.clf.predict(X, batch_size=32, verbose=0)[:,1] >= 0.5).astype(int)
        return preds

    def predict_proba(self, X):
        (X, _) = prepare_dataset_for_NN((X, None), self.img_shape, self.img_channels)
        pred_probs = self.clf.predict_proba(X, batch_size=32, verbose=0)
        return pred_probs

    def evaluate(self, X_test, y_test):
        (X_test, y_test) = prepare_dataset_for_NN((X_test, y_test), self.img_shape, self.img_channels)
        score = self.clf.evaluate(X_test, y_test, verbose = 0)
        print('Test score:', score[0])
