# K-NN Classifier

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

x_actual = np.linspace(np.amin(X_train[:, 0]), np.amax(X_train[:, 0]), 10).astype(int)
y_actual = np.linspace(np.amin(X_train[:, 1]), np.amax(X_train[:, 1]), 10).astype(int)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Creating confusion matrix for the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

x_scaled = np.linspace(np.amin(X_train[:, 0]), np.amax(X_train[:, 0]), 10)
y_scaled = np.linspace(np.amin(X_train[:, 1]), np.amax(X_train[:, 1]), 10)
# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
# To create a pixel grid for background with 0.01 steps
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# To colorize the pixel grid in either red of green, ravel is used to flatten the dataset
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# The for loop will loop twice: once to plot "will not buy" points with red color and
# the second time to plot "will buy" points with green color
my_labels = ["Will not buy", "Will buy"]
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = my_labels[j], edgecolors='black')
plt.title('KNN Classifier')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.xticks(x_scaled, x_actual)
plt.yticks(y_scaled, y_actual)
plt.legend()
plt.show()
