from keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from keras.models import Model, Sequential, load_model
from numpy import array, append, asarray
from numpy.random import rand
from matplotlib import pyplot as plt
from PIL import Image

image = Image.open("lena.bmp")
image.load()
im = asarray(image, dtype="B")

im = im / 256


model = Sequential()
model.add(Conv2D(3, 3, strides=1, activation="sigmoid", input_shape=(None, None, 3)))
model.add(Conv2D(6, 3, strides=1, activation="sigmoid", input_shape=(None, None, 3)))
model.add(Conv2D(3, 3, strides=1, activation="sigmoid", input_shape=(None, None, 3)))
model.add(Conv2DTranspose(3, 3, strides=1, activation="sigmoid", input_shape=(None, None, 3)))
model.add(Conv2DTranspose(6, 3, strides=1, activation="sigmoid", input_shape=(None, None, 3)))
model.add(Conv2DTranspose(3, 3, strides=1, activation="sigmoid", input_shape=(None, None, 3)))

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])





# data = rand(100, 10, 10, 3)
data = array([im for _ in range(200)])

model.fit(data, data, batch_size=40, epochs=10)
# model.save("decodercutie.model")

# model = load_model("decodercutie.model")
encoded = model.predict(array([im]))


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.imshow(im * 256)
ax2.imshow(encoded[0] * 256)

plt.show()
