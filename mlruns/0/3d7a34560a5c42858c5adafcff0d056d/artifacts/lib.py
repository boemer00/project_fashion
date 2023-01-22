import pandas as pd
import numpy as np
import argparse
import configparser
import mlflow
import mlflow.keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

class Fashion:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        # load data using tensoflow library
        (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()
        
    def transform_data(self):
        # reshape data to have a depth of 1
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)
        
        # convert y to categorical
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
    
    def get_data(self):
        # call the data splits
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        # create a CNN model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    

if __name__ == '__main__':
    
    # read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # get the parameters from the config file
    batch_size = config.getint('parameters', 'batch_size')
    epochs = config.getint('parameters', 'epochs')
    validation_split = config.getfloat('parameters', 'validation_split')

    # set up arguments to be parametrised from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./lib.py')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--validation_split', type=float, default=0.2)
    args = parser.parse_args()
    file_path = args.file_path
    
    # start an MLflow run
with mlflow.start_run():
   
    # log parameters
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('epochs', epochs)
    mlflow.log_param('validation_split', validation_split)

    # instatiate the fashion class 
    fashion = Fashion()

    # call functions
    fashion.load_data()
    fashion.transform_data()
    X_train, X_test, y_train, y_test = fashion.get_data()
    model = fashion.build_model()

    # train the model
    model.fit(X_train, 
              y_train, 
              batch_size=batch_size, 
              epochs=epochs, 
              validation_split=validation_split, 
              verbose=1)

    # evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # log metrics
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_acc", test_acc)

    # log the model
    mlflow.keras.log_model(model, "model")
    
    # log the file path
    mlflow.log_artifact(file_path)

mlflow.end_run()

    
    # # instatiate the fashion class 
    # fashion = Fashion()

    # # call functions
    # fashion.load_data()
    # fashion.transform_data()
    # X_train, X_test, y_train, y_test = fashion.get_data()
    # model = fashion.build_model()

    # # train the model
    # model.fit(X_train, 
    #           y_train, 
    #           batch_size=batch_size, 
    #           epochs=epochs, 
    #           validation_split=validation_split, 
    #           verbose=1)

    # # evaluate the model
    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print('Test accuracy:', test_acc)