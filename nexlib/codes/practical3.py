import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


train = pd.read_csv('./Handwritten/mnist_train.csv')
test = pd.read_csv('./Handwritten/mnist_test.csv')

X_train, y_train = train.iloc[:,1:].values/255.0, train.iloc[:,0]
X_test, y_test = test.iloc[:,1:].values/255.0, test.iloc[:,0]


def create_model(regularization=None, dropout=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularization),
        tf.keras.layers.Dropout(0.5) if dropout else tf.keras.layers.Activation('linear'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
history = model.fit(X_train, y_train, epochs=5, batch_size=64,
                    validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Without Regularization:", accuracy*100)

l2_model = create_model(tf.keras.regularizers.l2(0.001))
l2_history = l2_model.fit(X_train, y_train, epochs=5, batch_size=64,
                          validation_data=(X_test, y_test))

loss, accuracy = l2_model.evaluate(X_test, y_test)
print("With L2 Regularization:", accuracy*100)

dropout_model = create_model(dropout=True)
dropout_history = dropout_model.fit(X_train, y_train, epochs=5, batch_size=64,
                                    validation_data=(X_test, y_test))

loss, accuracy = dropout_model.evaluate(X_test, y_test)
print("With Dropout:", accuracy*100)


plt.plot(history.history['loss'], label='No Regularization')
plt.plot(l2_history.history['loss'], label='L2 Regularization')
plt.plot(dropout_history.history['loss'], label='Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.show()


plt.plot(history.history['val_accuracy'], label='No Regularization')
plt.plot(l2_history.history['val_accuracy'], label='L2 Regularization')
plt.plot(dropout_history.history['val_accuracy'], label='Dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()
plt.show()