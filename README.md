### Traffic Prediction in Kabul city, Afghanistan
This project amis to predict traffic situations in in Kabul city, Afghanistan using logistic regression model.

### Data
The data used in this project comes from website Kaggle.com
https://www.kaggle.com/datasets/hasibullahaman/traffic-prediction-dataset/data

### Model
The model that uses the sigmoid function to calculate the output.

```py
def sigmoid(self, z):
  return 1/(1+np.exp(-z))
```
The loss function of the model using binary cross entropy.
```py
def cross_entropy(self, x, y):
  eps = 1e-15  # Small constant value to prevent division by zero
  z = np.dot(self.w, x.T) + self.b
  y_pred = self.sigmoid(z)
  return -(np.dot(y.T,np.log(y_pred + eps)) + np.dot((1-y).T, np.log(1-y_pred + eps)))/x.shape[0] + self.regularization_cost()
```

The gradent of the cross entropy.
```py
z = np.dot(self.w, x_batch.T) + self.b
y_pred = self.sigmoid(z)

gred_w = np.dot(x_batch.T, (y_pred - y_batch))/num_samples + self.c*self.w
gred_b = np.sum(y_pred - y_batch)/num_samples
```

Update the weight of the model.
```
self.w = self.w - (self.learning_rete*gred_w)
self.b = self.b - (self.learning_rete*gred_b)
```
