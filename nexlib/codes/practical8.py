import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Load offline MNIST
train = pd.read_csv('Handwritten/mnist_train.csv').values
test = pd.read_csv('Handwritten/mnist_test.csv').values

X, y = train[:,1:] / 255.0, train[:,0]
Xt, yt = test[:,1:] / 255.0, test[:,0]

results = []

# Hyperparameter comparison
for lr, batch, units in [(0.01, 32, 64), (0.001, 64, 128)]:

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, y, epochs=5, batch_size=batch, verbose=0,
              callbacks=[tf.keras.callbacks.TensorBoard(
                  log_dir=f'logs/{lr}_{batch}')])

    loss, acc = model.evaluate(Xt, yt, verbose=0)
    results.append(acc)
    print(lr, batch, units, "Accuracy:", acc)

plt.bar(['Config 1', 'Config 2'], results)
plt.ylabel('Accuracy')
plt.title('Hyperparameter Comparison')
plt.show()


%load_ext tensorboard
%tensorboard --logdir logs