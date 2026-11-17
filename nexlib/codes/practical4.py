import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


train = pd.read_csv('./Handwritten/mnist_train.csv')
test = pd.read_csv('./Handwritten/mnist_test.csv')

X_train, y_train = train.iloc[:,1:].values/255.0, train.iloc[:,0]
X_test, y_test = test.iloc[:,1:].values/255.0, test.iloc[:,0]


def create_model(activation='relu'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


relu_model = create_model()
relu_history = relu_model.fit(X_train, y_train, epochs=5, batch_size=64,
                              validation_data=(X_test, y_test))
relu_loss, relu_accuracy = relu_model.evaluate(X_test, y_test)

sigmoid_model = create_model('sigmoid')
sigmoid_history = sigmoid_model.fit(X_train, y_train, epochs=5, batch_size=64,
                                    validation_data=(X_test, y_test))
sigmoid_loss, sigmoid_accuracy = sigmoid_model.evaluate(X_test, y_test)

tanh_model = create_model('tanh')
tanh_history = tanh_model.fit(X_train, y_train, epochs=5, batch_size=64,
                              validation_data=(X_test, y_test))
tanh_loss, tanh_accuracy = tanh_model.evaluate(X_test, y_test)

print("ReLU Accuracy:", relu_accuracy * 100)
print("Sigmoid Accuracy:", sigmoid_accuracy * 100)
print("Tanh Accuracy:", tanh_accuracy * 100)


plt.plot(relu_history.history['accuracy'], label='ReLU')
plt.plot(sigmoid_history.history['accuracy'], label='Sigmoid')
plt.plot(tanh_history.history['accuracy'], label='Tanh')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()
plt.show()


plt.plot(relu_history.history['loss'], label='ReLU')
plt.plot(sigmoid_history.history['loss'], label='Sigmoid')
plt.plot(tanh_history.history['loss'], label='Tanh')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.show()




def get_gradients(model):
    with tf.GradientTape() as tape:
        predictions = model(X_train[:64], training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_train[:64], predictions
        )
        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    return [tf.reduce_mean(tf.abs(g)).numpy() for g in gradients if g is not None]

plt.plot(get_gradients(relu_model), label='ReLU')
plt.plot(get_gradients(sigmoid_model), label='Sigmoid')
plt.plot(get_gradients(tanh_model), label='Tanh')
plt.xlabel('Layer')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Flow Comparison')
plt.legend()
plt.show()