import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 


df = pd.read_csv("./Dataset/IMDB/IMDB Dataset.csv")

X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(200,)),
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)


plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')

plt.title("LSTM Model Accuracy")
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()


loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss: ", loss) 
print("Test Accuracy: ", accuracy * 100, "%") 


predictions = model.predict(X_test[:10])

for i in range(10):
    actual = "Positive" if y_test.iloc[i] == 1 else "Negative"
    predicted = "Positive" if predictions[i][0] >= 0.5 else "Negative" 

    print("Review: ", i+1)
    print("Actual Sentiment: ", actual)
    print("Predicted Sentiment: ", predicted)
    print("----------------------")