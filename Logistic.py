#logistic regression using keras
import pandas as pd
import numpy as np
from tensorflow import Variable
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt


# load dataset
data = pd.read_csv("heart_data.csv")
x = data.drop(['target'], axis=1)
y = data['target']
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

num_of_classes = 1
num_of_features = 13

model = Sequential()

step = Variable(0, trainable=False)
boundaries = [50]
values = [1e-2, 1e-3]
learning_rate_fn = optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
optimizer = optimizers.Adam(lr=learning_rate_fn(step))

model.add(Dense(num_of_classes, input_dim=num_of_features, kernel_initializer = HeNormal()))
#model.add(BatchNormalization())
model.add(Activation("sigmoid"))
#model.add(Dropout(0.2))
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=600, batch_size = 1025, validation_data=(x_test, y_test))
y_pred = model.predict(x_test).round()

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("F1 Score:", round(f1_score(y_test, y_pred), 2))
print("AUC of ROC: ", round(roc_auc_score(y_test, y_pred), 2))

plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend(['training', 'validation'], loc='upper left')
plt.grid()
plt.yticks(np.arange(0.3, 1, step=0.05))

plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend(['training', 'validation'], loc='upper left')
plt.grid()

#gets weights of the model
weights = model.get_layer(index=0).get_weights()[0]

#make a bar graph of weights vs feature
plt.figure(3)
plt.bar(data.columns[:13], np.reshape(np.absolute(weights), (13,)))
plt.title("|Weights| vs Features")
plt.show()