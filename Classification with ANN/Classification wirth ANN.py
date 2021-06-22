#---------------- CLASSIFICATION WITH ANN ----------------#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


#---------------- DATASET ----------------#
'''
Data set:
RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,
Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
'''

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = OrdinalEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1].reshape(10000,1)).reshape(10000)
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]


#---------------- CREATE TESTING AND TRAINING DATASET ----------------#
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#---------------- CREATING ARTIFICIAL NEURAL NETWORK ----------------#
# Initialising the Artificial Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#---------------- TRAIN THE MODEL ----------------#
# Compiling the Artificial Neural Network
'''
Optimizer : ADAM
Loss: Binary Cross Entropy. We have only two classes for the classification task.
Mertics: Accuracy. We need to calculates how often predictions equal labels.
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the Artificial Neural Network to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


#---------------- TEST THE MODEL ----------------#
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
x_predicted =  np.where(y_pred==True)[0]
print('Loyal Clients: ', x_predicted)


#---------------- GENERATING CONFUSION MATRIX ----------------#
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True)
plt.show()