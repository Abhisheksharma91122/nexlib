import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt


train = pd.read_csv('./Handwritten/mnist_train.csv')
test = pd.read_csv('./Handwritten/mnist_test.csv')

X_train, y_train = train.iloc[:,1:].values/255.0, train.iloc[:,0]
X_test, y_test = test.iloc[:,1:].values/255.0, test.iloc[:,0]


# MLP Model
mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

mlp.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

mlp_history = mlp.fit(X_train, y_train, epochs=5, batch_size=64,
                      validation_data=(X_test, y_test))


# Reshape data for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Data Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1)
])

# CNN Model
cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    augmentation,
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn_history = cnn.fit(X_train, y_train, epochs=5, batch_size=64,
                      validation_data=(X_test, y_test))

mlp_loss, mlp_accuracy = mlp.evaluate(X_test.reshape(-1,784), y_test)
cnn_loss, cnn_accuracy = cnn.evaluate(X_test, y_test)

print("MLP Accuracy:", mlp_accuracy * 100, "%")
print("CNN Accuracy:", cnn_accuracy * 100, "%")

# Accuracy Comparison
plt.plot(mlp_history.history['val_accuracy'], label='MLP')
plt.plot(cnn_history.history['val_accuracy'], label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MLP vs CNN Accuracy')
plt.legend()
plt.show()


# Loss Comparison
plt.plot(mlp_history.history['val_loss'], label='MLP')
plt.plot(cnn_history.history['val_loss'], label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP vs CNN Loss')
plt.legend()
plt.show()