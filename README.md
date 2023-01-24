# Project Fashion

This project is a simple implementation of a Convolutional Neural Network (CNN) to classify images of clothing from the Fashion-MNIST dataset.

![](fashion.png)

## Key Requirements
- Matplotlib
- MLflow
- Numpy
- Pandas
- Python 3.10.6
- Scikit-learn
- Tensorflow

## Installation
You can clone this repository using git and then donwload the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset:

```$ git clone https://github.com/boemer00/project_fashion.git```

## Usage
The main script is lib.py which contains the Fashion class. The class provides the following functions:

- **load_data()**: loads the data from the Fashion-MNIST dataset and splits it into train and test sets.
- **transform_data()**: normalises and reshapes the data, and converts the labels to categorical format.
- **get_data()**: returns the train and test sets.
- **build_model()**: builds and returns a CNN model.
You can use the class to train and evaluate the model by running:

```python lib.py```

This script will load the data, preprocess it, train the model and evaluate it on the test set. It will also save the model in the models folder.

## Evaluation
The script will output the accuracy of the model on the test set.
If you want to run the unit test cases, use the command:

```python -m unittest test_fashion.py```

## Note
The script is using a fixed *random_state* (42) for splitting the data and for the model initialisation, if you want to change the random_state, you can do it by passing the parameter to the Fashion class, inside the load_data function.

## Results
The Confusion Matrix below shows that the model predict correct **98.9%** of all `Trouser` images. However, the model was not so successful in predicting `Shirt`, achieving **84.1%** accuracy. One possible explanation to such a high accuracy is that trousers have a unique shape, whereas shirts have similar shapes to T-shirt/Top and Coat. This is interesting because it could explain the high numbers of misclassified images.

![](confusion_matrix_renato_boemer.png)

## Contributions
Pull requests and bug reports are welcome :)

## License
This project is licensed under the [MIT License](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

## Acknowledgments
This project is inspired by the TensorFlow tutorials and the Fashion-MNIST dataset.
