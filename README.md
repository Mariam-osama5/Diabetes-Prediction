# Diabetes Prediction - Machine Learning Project

## Overview
This project aims to predict whether a person is diabetic using medical measurements.  
The model is trained on the **PIMA Indians Diabetes Dataset** using several ML algorithms and evaluated to find the best one.

---

## Steps Performed

1. **Data Exploration (EDA)**
   - Check data shape, types, and missing values
   - Visualize distributions and correlations

2. **Data Cleaning**
   - Replaced invalid 0s in key columns with `NaN`
   - Filled missing values with median

3. **Preprocessing**
   - Feature scaling using StandardScaler

4. **Model Building**
   - Tried: KNN, Logistic Regression, Decision Tree, Random Forest
   - Evaluated with Accuracy, Precision, Recall, F1-Score

5. **Best Model**
   - KNN gave the best results after tuning hyperparameters

---

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn

---

## Future Improvements
- Hyperparameter tuning with GridSearchCV
- Deploy the model using Streamlit or Flask
- Add a user interface for non-technical users

---

## 📎 Dataset
Available on Kaggle: [PIMA Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
