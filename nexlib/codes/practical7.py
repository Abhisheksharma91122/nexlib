import tensorflow as tf
import matplotlib.pyplot as plt

train = tf.keras.utils.image_dataset_from_directory(
    './cats_and_dogs_filtered/train',
    image_size=(150,150), batch_size=32
)

test = tf.keras.utils.image_dataset_from_directory(
    './cats_and_dogs_filtered/validation',
    image_size=(150,150), batch_size=32
)


# Load pretrained VGG16
base_model = tf.keras.applications.VGG16(
    weights='imagenet', include_top=False,
    input_shape=(150,150,3)
)

# Freeze pretrained layers
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150,150,3)),
    tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train with frozen layers
history = model.fit(train, epochs=3, validation_data=test)

loss, accuracy = model.evaluate(test)
print("Frozen Model Accuracy:", accuracy*100)


# Fine-tune last VGG16 block
base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = layer.name.startswith('block5')

model.compile(optimizer=tf.keras.optimizers.Adam(0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(train, epochs=3, validation_data=test)

loss, accuracy = model.evaluate(test)
print("Fine-tuned Model Accuracy:", accuracy*100)


# Accuracy comparison
plt.plot(history.history['val_accuracy'], label='Frozen')
plt.plot(history2.history['val_accuracy'], label='Fine-tuned')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Loss comparison
plt.plot(history.history['val_loss'], label='Frozen')
plt.plot(history2.history['val_loss'], label='Fine-tuned')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()