import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils

if __name__ == '__main__':
    # Some code has been adapted from https://nextjournal.com/gkoehler/digit-recognition-with-keras

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    (x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()

    x_train = x_train_raw.copy()
    x_test = x_test_raw.copy()

    #show some of the images
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(x_test[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(y_test[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # building the input vector from the 28x28 pixels
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalizing the data to help with the training
    x_train /= 255
    x_test /= 255

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.1)
    model.save('mnist_classifier')
    # model = load_model('mnist_classifier')
    score = model.evaluate(x_test, y_test, verbose = 0)

    print(score)

    # show some of the images and predict them with the trained model
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(x_test_raw[i], cmap='gray', interpolation='none')
        # result comes back as an array containing prediction for all classes
        # therefore we must do some work to extract the most likely class
        one_image = np.expand_dims(x_test[i], axis=0) # one dimensions must be added because we are using batches
        result_array = model.predict(one_image)
        max_result = 0
        predicted_number = 0
        for index, result in enumerate(result_array[0]):
            if result > max_result:
                max_result = result
                predicted_number = index
        plt.title("Predicted Digit: {}".format(predicted_number))
        plt.xticks([])
        plt.yticks([])
    plt.show()
