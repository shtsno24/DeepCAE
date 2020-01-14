from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, SeparableConv2D
from tensorflow.compat.v1.keras.initializers import he_normal
# from tensorflow.keras.initializers import he_normal
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import save_img
import matplotlib.pyplot as plt
import numpy as np


batch_size = 128
# num_classes = 10
epochs = 1000

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("image_data_format : " + K.image_data_format())

# x_train = x_train.astype('uint8')
# x_test = x_test.astype('uint8')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(SeparableConv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 kernel_initializer='he_normal'))

model.add(UpSampling2D(size=(2, 2)))
model.add(SeparableConv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 kernel_initializer='he_normal'))

model.add(UpSampling2D(size=(2, 2)))
model.add(SeparableConv2D(1, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 kernel_initializer='he_normal'))

# model.compile(loss=keras.losses.categological_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
# model.compile(loss=keras.losses.binary_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

fit = model.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, x_test))
scores = model.evaluate(x_test, x_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[0 + 1])

model.summary()
model.save("keras_mnist_DCAE.h5")
model.save_weights("keras_mnist_DCAE_weight.h5")
model_json = model.to_json()
with open("keras_mnist_DCAE.json", "w") as f:
    f.write(model_json)


def representative_dataset_gen():
    for i in range(10):
        yield [x_train[i: i + 1]]


# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.representative_dataset = representative_dataset_gen
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# tflite_model = converter.convert()
# with open("tflite_mnist_DCAE.tflite", "wb") as f:
#     f.write(tflite_model)

# x_test = x_test.astype('uint8')
input_img = np.array([[x for x in range(28)] for y in range(28)], dtype=np.float32).reshape(x_test[0:1].shape)
predict_img = model.predict(input_img)
# print(input_img[0].transpose(2, 0, 1))
# print(predict_img[0].transpose(2, 0, 1))

input_img = x_test[0:1]
predict_img = model.predict(input_img)
input_img = input_img[0]
predict_img = predict_img[0]
# print(array_to_img(input_img).shape, array_to_img(predict_img).shape)
# save_img("input.png", input_img)
# save_img("predict.png", predict_img)

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="training")
    axL.plot(fit.history['val_loss'],label="validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="training")
    axR.plot(fit.history['val_acc'],label="validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig('./history.png')
plt.close()