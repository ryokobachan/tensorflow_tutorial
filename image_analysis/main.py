import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from model import make_model

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

test_images = test_images / 255.0

model = make_model()

model.load_weights('saved_weight/saved_weight')
model.summary()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
