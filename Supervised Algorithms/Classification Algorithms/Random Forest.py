# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import Data
Data = pd.read_csv("D:\D\Internship (Tectonas)\Programs\diabetes.csv")
print(Data.head(5))


# Spliting features
# Features
X = Data.iloc[:,0:8]
# Target variable
y = Data.Outcome


# Dividing train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# Building a model
clf = RandomForestClassifier()
clf = clf.fit(X_train,y_train)


# Predict
y_pred = clf.predict(X_test)


# Classification metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

