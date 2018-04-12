from keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from keras.models import Model, Sequential, load_model
from numpy import array, append, asarray
from numpy.random import rand
from matplotlib import pyplot as plt
from PIL import Image

image = Image.open("lena.bmp")
image.load()
im = asarray(image, dtype="B")

i = im / 256


model = Sequential()
model.add(Conv2D(6, 4, strides=4, activation="sigmoid", input_shape=(None, None, 3)))
model.add(Conv2DTranspose(3, 4, strides=4, activation="sigmoid"))

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])





# data = rand(100, 10, 10, 3)
data = array([i for _ in range(200)])

model.fit(data, data, batch_size=1, epochs=40)
model.save("decodercutie.model")

# model = load_model("decodercutie.model")
encoded = model.predict(array([i]))

print(encoded)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.imshow(im)
ax2.imshow(encoded[0] * 256)

plt.show()
