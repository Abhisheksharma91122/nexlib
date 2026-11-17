import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


train = tf.keras.utils.image_dataset_from_directory(
    'cats_and_dogs_filtered/train', image_size=(160,160), batch_size=32)

val = tf.keras.utils.image_dataset_from_directory(
    'cats_and_dogs_filtered/validation', image_size=(160,160), batch_size=32)

train = train.map(lambda x,y: (preprocess_input(x),y)).prefetch(tf.data.AUTOTUNE)
val = val.map(lambda x,y: (preprocess_input(x),y)).prefetch(tf.data.AUTOTUNE)


base = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(160,160,3))

base.trainable = False

model = tf.keras.Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

h1 = model.fit(train, validation_data=val, epochs=3)


base.trainable = True

for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

h2 = model.fit(train, validation_data=val, epochs=3)


plt.plot(h1.history['accuracy'] + h2.history['accuracy'], label='Training Accuracy')
plt.plot(h1.history['val_accuracy'] + h2.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(2.5, linestyle='--', label='Fine-Tuning Starts')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ResNet50: Frozen vs Fine-Tuned')
plt.legend()
plt.grid()
plt.show()


model.evaluate(val)