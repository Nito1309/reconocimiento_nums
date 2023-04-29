import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math

ingresados, datosMeta = tfds.load(
    "mnist", as_supervised=True, with_info=True)

datos, pruebas = ingresados["train"], ingresados["test"]
clases = datosMeta.features["label"].names


def normalize(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas


datos = datos.map(normalize)
pruebas = datos.map(normalize)

plt.figure(figsize=(10, 10))

for i, (imagen, etiqueta) in enumerate(datos.take(25)):
    imagen = imagen.numpy().reshape((28, 28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(clases[etiqueta])
plt.show()

mdl = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

mdl.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

entrenamiento = datosMeta.splits["train"].num_examples
num = datosMeta.splits["test"].num_examples

TAMANO_LOTE = 32

datos = datos.repeat().shuffle(
    entrenamiento).batch(TAMANO_LOTE)
pruebas = pruebas.batch(TAMANO_LOTE)


historial = mdl.fit(
    datos,
    epochs=60,
    steps_per_epoch=math.ceil(entrenamiento/TAMANO_LOTE)
)

mdl.save('numeros_regular.h5')

mdl_convu = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(
        28, 28, 1), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(
        28, 28, 1), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dense(units=50, activation="relu"),
    tf.keras.layers.Dense(units=50, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

mdl_convu.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
