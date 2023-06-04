

import pandas as pd
import time
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.preprocessing import StandardScaler  # For normalizing the data
from sklearn.metrics import confusion_matrix  # For creating the confurion matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
# For Deep learning
from tensorflow import keras
from tensorflow.keras import layers


"""
 For the performance of an algorithm, we need Big O notation or big omega notation. 
 Time is not an indicator for performance of an algorithm. However in this experiment
 we used time as an indicator that shows the performance. That is why it will be hard to calculate each 
 algortihm's performance with Big O notation or big omega notation
"""
# This function gives us the time value in miliseconds.


def current_milli_time():
    return round(time.time() * 1000)

# This function calculates the accuracy rate of a confusion matrix.


def calculate_AR(confusion_matrix):
    return (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])


def print_values(algorithm: str, accuracy_rate: float, training_time: int):
    print(algorithm, "Accuracy rate", accuracy_rate,
          "Training time:", training_time)


# Obtaining dataset with malicious mote, Please tahe attention that the files must be
# in the same folder with this python file.
# flnme1 = "DR-9N1M1R.csv"
# # Obtaining dataset with normal motes
# flnme2 = "DR-10N1R.csv"
df = pd.read_csv("merged.csv")


# Source and destination IP addresses are extracted from the datasets before the normalization process.
X = df.iloc[:, 3:16].values
y = df.iloc[:, 16:17].values.ravel()

# The dataset was split into test and training datasets in the amount of 2/3. (2/3 training, 1/3 testing).
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)


# Normalizing the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Scripts for Logistic regression
lrstart_time = current_milli_time()  # Obtaining initial time of the training
# Creating the logistic regression object
logr = LogisticRegression(random_state=0)
logr.fit(x_train, y_train)  # Training the data
lrend_time = current_milli_time()  # Obtaining ending time of the training
LRduration = lrend_time - lrstart_time  # Calculating the duration
y_pred_lr = logr.predict(x_test)  # Predicting data
cm_lr = confusion_matrix(y_test, y_pred_lr)  # Creating confusion matrix
ar_lr = calculate_AR(cm_lr)  # Calculating accuracy rate.


# Scripts for Random Forest Classifation
rfstart_time = current_milli_time()  # Obtaining initial time of the training
# Creating the Random Forest Classifation object
rfc = RandomForestClassifier(n_estimators=8, criterion='entropy')
rfc.fit(x_train, y_train)  # Training the data
rfend_time = current_milli_time()  # Obtaining ending time of the training
RFduration = rfend_time - rfstart_time  # Calculating the duration
y_pred_rfc = rfc.predict(x_test)  # Predicting data
cm_rfc = confusion_matrix(y_test, y_pred_rfc)  # Creating confusion matrix
ar_rfc = calculate_AR(cm_rfc)  # Calculating accuracy rate.


# Scripts for Decision Tree Classifier
dtstart_time = current_milli_time()  # Obtaining initial time of the training
# Creating the Decision Tree Classifier object
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)  # Training the data
dtend_time = current_milli_time()  # Obtaining ending time of the training
DTduration = dtend_time - dtstart_time  # Calculating the duration
y_pred_dtc = dtc.predict(x_test)  # Predicting data
cm_dtc = confusion_matrix(y_test, y_pred_dtc)  # Creating confusion matrix
ar_dtc = calculate_AR(cm_dtc)  # Calculating accuracy rate.


# Scripts for Naive Bayes Classifier
nbstart_time = current_milli_time()  # Obtaining initial time of the training
gnb = GaussianNB()  # Creating the Naive Bayes Classifier object
gnb.fit(x_train, y_train)  # Training the data
nbend_time = current_milli_time()  # Obtaining ending time of the training
NBduration = nbend_time - nbstart_time  # Calculating the duration
y_pred_nb = gnb.predict(x_test)  # Predicting data
cm_nb = confusion_matrix(y_test, y_pred_nb)  # Creating confusion matrix
ar_nb = calculate_AR(cm_nb)  # Calculating accuracy rate.


# Scripts for KNN Classifier
knnstart_time = current_milli_time()  # Obtaining initial time of the training
knn = KNeighborsClassifier()  # Creating the KNN Classifier object
knn.fit(x_train, y_train)  # Training the data
knnend_time = current_milli_time()  # Obtaining ending time of the training
knnduration = knnend_time - knnstart_time  # Calculating the duration
y_pred_knn = knn.predict(x_test)  # Predicting data
cm_knn = confusion_matrix(y_test, y_pred_knn)  # Creating confusion matrix
ar_knn = calculate_AR(cm_knn)  # Calculating accuracy rate.


# Scripts for Deep learning
# Creating the Deep learning object
# We established 13 input, that is why we have 13 columns. the other layers are established with 50, 100, 300, 100, 50, 1 layers repectively.
model = keras.Sequential(
    [
        keras.Input(shape=(13)),
        layers.Dense(50, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(300, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
start_time = current_milli_time()  # Obtaining initial time of the training
model.compile(optimizer="Nadam", loss="binary_crossentropy",
              metrics=['binary_accuracy'])
model.fit(x_train, y_train, epochs=60)  # Training the data
end_time = current_milli_time()  # Obtaining ending time of the training
DLduration = end_time - start_time  # Calculating the duration
y_pred_dl = model.predict(x_test)  # Predicting data
y_pred_dl = (y_pred_dl > 0.7)
cm_dl = confusion_matrix(y_test, y_pred_dl)  # Creating confusion matrix
ar_dl = calculate_AR(cm_dl)  # Calculating accuracy rate.

# Printing the results
print("")
print_values("Logistic Regression", ar_lr, LRduration)
print_values("Random Forest Classifation", ar_rfc, RFduration)
print_values("Decision Tree Classifier", ar_dtc, DTduration)
print_values("Naive Bayes Classifier", ar_nb, NBduration)
print_values("KNN Classifier", ar_knn, knnduration)
print_values("Deep Learning", ar_dl, DLduration)

#Handling class imbalance
smote = SMOTE(random_state=0)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# Hyperparameter tuning for Random Forest
rfc = RandomForestClassifier(random_state=0)

param_grid = {
    'n_estimators': [8, 10, 12, 15, 20],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rfc = GridSearchCV(
    estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
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
print("Random Forest Classifation Cross Validation Score: ",
      rfc_best_cross_val_score)


"""
Also the Fuzzy Pattern Tree Classifier was tested for experiments.
fptstart_time=current_milli_time()
from fylearn.fpt import FuzzyPatternTreeClassifier     
fpc4=FuzzyPatternTreeClassifier()
fpc4.fit(x_train,y_train)
y_pred_fpc4 = fpc4.predict(x_test)

cm_fpc4=confusion_matrix(y_test,y_pred_fpc4)

fptend_time=current_milli_time()
FPTduration=fptend_time-fptstart_time

"""
