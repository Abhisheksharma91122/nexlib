import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

train = pd.read_csv('./Handwritten/mnist_train.csv')
test = pd.read_csv('./Handwritten/mnist_test.csv')

X_train, y_train = train.iloc[:,1:].values/255.0, train.iloc[:,0]
X_test, y_test = test.iloc[:,1:].values/255.0, test.iloc[:,0]


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64,
                    validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy*100, 4))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Traning and validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()