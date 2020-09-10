import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from model import make_model

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = make_model()

model.fit(train_images, train_labels, epochs=30)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.save('saved_model')
model.save_weights('saved_weight/saved_weight')
model.summary()
