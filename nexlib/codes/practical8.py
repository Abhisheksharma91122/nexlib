import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Load offline MNIST dataset
train = pd.read_csv('./Handwritten/mnist_train.csv')
test = pd.read_csv('./Handwritten/mnist_test.csv')

X_train, y_train = train.iloc[:,1:].values/255.0, train.iloc[:,0]
X_test, y_test = test.iloc[:,1:].values/255.0, test.iloc[:,0]

# Create model
def create_model(lr, layers):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(784,))])
    for i in range(layers):
        model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Different configurations
configs = [
    (0.001, 32, 1),
    (0.01, 32, 2),
    (0.001, 64, 2)
]

results = []

for i, (lr, batch, layers) in enumerate(configs):
    model = create_model(lr, layers)

    log_dir = f'logs/config_{i+1}'
    callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(X_train, y_train, epochs=5, batch_size=batch,
              validation_split=0.2, callbacks=[callback], verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    results.append([lr, batch, layers, accuracy*100])

# Compare results
for r in results:
    print(f"LR: {r[0]}, Batch: {r[1]}, Layers: {r[2]}, Accuracy: {r[3]:.2f}%")



%load_ext tensorboard
%tensorboard --logdir logs