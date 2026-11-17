#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt


# In[13]:


# Input data (AND gate)
X = np.array([
 [0,0],
 [0,1],
 [1,0],
 [1,1]
])

# Expected output
y = np.array([0,0,0,1])

print("Input Data:\n", X)
print("Output Data:", y)


# In[14]:


class Perceptron:
  def __init__(self):
    self.w = np.zeros(2)
    self.b = 0
    self.lr = 0.1

  def activation(self, x):
    if x >= 0:
      return 1
    else:
      return 0

  def fit(self, X, y):
    for epoch in range(30):
      for i in range(len(X)):
        output = np.dot(self.w, X[i]) + self.b
        prediction = self.activation(output)
        error = y[i] - prediction

        self.w += self.lr * error * X[i]
        self.b += self.lr * error

  def predict(self, X):
   result = []

   for x in X:
    output = np.dot(self.w, x) + self.b
    result.append(self.activation(output))

   return np.array(result)


# In[15]:


model = Perceptron()

model.fit(X, y)

print("Training Completed!")
print("Weights: ", model.w)
print("Bias!", model.b)


# In[16]:


predictions = model.predict(X)

print("Predicted Output: ", predictions)
print("Actual Output: ", y)

accuracy = np.mean(predictions == y) * 100

print("Accuracy: ", accuracy, "%")


# In[17]:


# Decision boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b) / w2
x_vals = np.array([-0.5, 1.5])
y_vals = -(model.w[0] * x_vals + model.b) / model.w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="black", s=100)
plt.plot(x_vals, y_vals, 'g-', label="Decision Boundary")


plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("AND Gate Decision Boundary")
plt.legend()
plt.grid()
plt.show()


# In[18]:


print("Final Weights:", model.w)
print("Final Bias:", model.b)


# In[14]:




