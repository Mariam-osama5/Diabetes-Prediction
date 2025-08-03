import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

##2-data colection
df = pd.read_csv(r"C:\Users\New\Desktop\archive\diabetes.csv")
print("shape:", df.shape)
print(df.head())
print(df.info())

##3-EDA + cleaning
pd.set_option('display.max_columns', None)
print(df.describe())
print("columns:", df.columns.tolist())
print("missing values:", df.isnull().sum())

##4- data preprocissing
colm_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in colm_with_zeros:
    df[col] = df[col].replace(0, np.nan)

for col in colm_with_zeros:
    median = df[col].median()
    df[col].fillna(median, inplace=True)

##5-model selection + training
x = df.drop('Outcome', axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

##6- model evaluation

#logistic regression
#make the model
model = LogisticRegression(max_iter=1000)
#train the model
model.fit(x_train, y_train)
#prediction
y_pred = model.predict(x_test)
#performance 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("confution matrix:", confusion_matrix(y_test, y_pred))
print("classification report:", classification_report(y_test, y_pred))

#random forest
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_score = rf_model.score(x_test, y_test)

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
knn_score = knn_model.score(x_test, y_test)

# SVM
svm_model = SVC()
svm_model.fit(x_train, y_train)
svm_score = svm_model.score(x_test, y_test)

#try the difference
print("Random Forest Accuracy:", rf_score)
print("KNN Accuracy:", knn_score)
print("SVM Accuracy:", svm_score)

#the best is KNN

## 7- model training
final_model = KNeighborsClassifier()
final_model.fit(x_train, y_train)

#Predict on test set
y_test_pred = final_model.predict(x_test)

#Evaluate
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
