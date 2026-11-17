import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("./Dataset/IMDB/IMDB Dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'].map({'positive':1, 'negative':0}),
    test_size=0.2, random_state=42
)


vocab_size = 10000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

x_train = tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(X_train), maxlen=250
)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(X_test), maxlen=250
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.LSTM(100, dropout=0.3),
    tf.keras.layers.Dense(1, activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(0.001))
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                    validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy * 100, "%")



# Accuracy Graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Example Predictions
predictions = model.predict(x_test[:5])

for i in range(5):
    print("Review:", X_test.iloc[i][:100])
    print("Actual:", "Positive" if y_test.iloc[i] == 1 else "Negative")
    print("Predicted:", "Positive" if predictions[i][0] >= 0.5 else "Negative")
    print()