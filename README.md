# Classification Algorithm Guide

This guide provides instructions on how to use the classification algorithm implemented in the provided code. The algorithm aims to classify datasets into two classes: normal motes (class 0) and malicious motes (class 1).

## Prerequisites

Before running the algorithm, make sure you have the following:

- Python installed (version 3.6 or higher)
- Required Python packages: pandas, sklearn, imblearn, tensorflow

## Step 1: Data Preparation

1. Place your dataset files in the same folder as the Python code file.

Certainly! Here are the updated code snippets with additional explanations:

## Step 2: Data Cleaning and Feature Extraction

The first part of the code performs data cleaning and feature extraction on the dataset. It includes the following steps:

```markdown
# Dataset import

df = pd.read_csv("Results/merged.csv")

# Source and destination IP addresses are extracted from the datasets before the normalization process.

X = df.iloc[:, 3:16].values
y = df.iloc[:, 16:17].values.ravel()

# The dataset was split into test and training datasets in the amount of 2/3. (2/3 training, 1/3 testing).

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Normalizing the data

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```

## Step 3: Running the Classification Algorithm

The second part of the code implements various classification algorithms and evaluates their performance. Each algorithm is trained on the training set and tested on the testing set. The algorithms implemented are as follows:

### Logistic Regression (LR)

```python
# Creating the logistic regression object
logr = LogisticRegression(random_state=0)
logr.fit(x_train, y_train)  # Training the data
y_pred_lr = logr.predict(x_test)  # Predicting data
cm_lr = confusion_matrix(y_test, y_pred_lr)  # Creating confusion matrix
ar_lr = calculate_AR(cm_lr)  # Calculating accuracy rate.
```

### Random Forest Classification (RFC)

```python
# Creating the Random Forest Classification object
rfc = RandomForestClassifier(n_estimators=8, criterion='entropy')
rfc.fit(x_train, y_train)  # Training the data
y_pred_rfc = rfc.predict(x_test)  # Predicting data
cm_rfc = confusion_matrix(y_test, y_pred_rfc)  # Creating confusion matrix
ar_rfc = calculate_AR(cm_rfc)  # Calculating accuracy rate.
```

### Decision Tree Classifier (DTC)

```python
# Creating the Decision Tree Classifier object
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)  # Training the data
y_pred_dtc = dtc.predict(x_test)  # Predicting data
cm_dtc = confusion_matrix(y_test, y_pred_dtc)  # Creating confusion matrix
ar_dtc = calculate_AR(cm_dtc)  # Calculating accuracy rate.
```

### Naive Bayes Classifier (NBC)

```python
# Creating the Naive Bayes Classifier object
gnb = GaussianNB()
gnb.fit(x_train, y_train)  # Training the data
y_pred_nb = gnb.predict(x_test)  # Predicting data
cm_nb = confusion_matrix(y_test, y_pred_nb)  # Creating confusion matrix
ar_nb = calculate_AR(cm_nb)  # Calculating accuracy rate.
```

### K-Nearest Neighbors Classifier (KNN)

```python
# Creating the KNN Classifier object
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)  # Training the data
y_pred_knn = knn.predict(x_test)  # Predicting data
cm_knn = confusion_matrix(y_test, y_pred_knn)  # Creating confusion matrix


ar_knn = calculate_AR(cm_knn)  # Calculating accuracy rate.
```

### Deep Learning (DL)

```python
# Creating the Deep learning object
model = keras.Sequential([
    keras.Input(shape=(13)),
    layers.Dense(50, activation="relu"),
    layers.Dense(100, activation="relu"),
    layers.Dense(300, activation="relu"),
    layers.Dense(100, activation="relu"),
    layers.Dense(50, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="Nadam", loss="binary_crossentropy", metrics=['binary_accuracy'])
model.fit(x_train, y_train, epochs=60)  # Training the data
y_pred_dl = model.predict(x_test)  # Predicting data
y_pred_dl = (y_pred_dl > 0.7)
cm_dl = confusion_matrix(y_test, y_pred_dl)  # Creating confusion matrix
ar_dl = calculate_AR(cm_dl)  # Calculating accuracy rate.
```

## Step 4: Handling Class Imbalance (Optional)

If your dataset has class imbalance, you can apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes. The code includes a section for handling class imbalance using SMOTE.

```python
# Handling class imbalance
smote = SMOTE(random_state=0)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
```

## Step 5: Hyperparameter Tuning (Optional)

The code includes an example of hyperparameter tuning for the Random Forest Classifier (RFC) using GridSearchCV.

```python
# Hyperparameter tuning for Random Forest
rfc = RandomForestClassifier(random_state=0)

param_grid = {
    'n_estimators': [8, 10, 12, 15, 20],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search_rfc.fit(x_train_res, y_train_res)

# Get the best parameters
best_params = grid_search_rfc.best_params_

# Train again with the best parameters
rfc_best = RandomForestClassifier(**best_params)
rfc_best.fit(x_train_res, y_train_res)

y_pred_rfc_best = rfc_best.predict(x_test)
cm_rfc_best = confusion_matrix(y_test, y_pred_rfc_best)
ar_rfc_best = calculate_AR(cm_rfc_best)

# Calculate cross validation score
rfc_best_cross_val_score = cross_val_score(rfc_best, X, y, cv=5).mean()

print("Random Forest Classifation Best Params Accuracy rate", ar_rfc_best)
print("Random Forest Classifation Cross Validation Score: ", rfc_best_cross_val_score)
```

## Additional Algorithm (Optional)

The code also includes an example of the Fuzzy Pattern Tree Classifier (FPT). This classifier is commented out in the code, but you can uncomment it to test the algorithm and compare its performance.

```python
"""
Also, the Fuzzy Pattern Tree Classifier was tested for experiments.
fptstart_time = current_milli_time()
from fylearn.fpt import FuzzyPatternTreeClassifier
fpc4 = FuzzyPatternTreeClassifier()
fpc4.fit(x_train, y_train)
y_pred_fpc4 = fpc4.predict(x_test)

cm_fpc4 = confusion_matrix(y_test, y_pred_fpc4)

f

ptend_time = current_milli_time()
FPTduration = fptend_time - fptstart_time
"""
```

## Credits

The project code in this repository is adapted from the following source:

[https://github.com/mukiraz/Detecting-RPL-Attacks](https://github.com/mukiraz/Detecting-RPL-Attacks)

**Please refer to the original repository for more details and additional resources.**

Once you run the code, you will see the accuracy rates, training times, and confusion matrices for each algorithm.

Feel free to customize and modify the code as needed for your specific dataset and requirements.

**🚀 Happy classifying!**
