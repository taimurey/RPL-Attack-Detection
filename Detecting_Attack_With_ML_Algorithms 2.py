import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from tensorflow import keras
from tensorflow.keras import layers

# Function to obtain the current time in milliseconds


def current_milli_time():
    return round(time.time() * 1000)

# Function to calculate the accuracy rate of a confusion matrix


def calculate_AR(confusion_matrix):
    return (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])

# Function to print the values in a beautiful format


def print_values(algorithm: str, accuracy_rate: float, training_time: int):
    print(f"{algorithm:<30} Accuracy rate: {accuracy_rate:<10} Training time: {training_time}")


df = pd.read_csv("merged.csv")
X = df.iloc[:, 3:16].values
y = df.iloc[:, 16:17].values.ravel()
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Store accuracy rates and training times in a dictionary
models = {
    "Logistic Regression": {"model": LogisticRegression(random_state=0), "duration": None, "accuracy": None},
    "Random Forest Classification": {"model": RandomForestClassifier(n_estimators=8, criterion='entropy'), "duration": None, "accuracy": None},
    "Decision Tree Classifier": {"model": DecisionTreeClassifier(criterion='entropy'), "duration": None, "accuracy": None},
    "Naive Bayes Classifier": {"model": GaussianNB(), "duration": None, "accuracy": None},
    "KNN Classifier": {"model": KNeighborsClassifier(), "duration": None, "accuracy": None},
    "Deep Learning": {"model": None, "duration": None, "accuracy": None}
}

# Logistic Regression
lr_start_time = current_milli_time()
logr = LogisticRegression(random_state=0)
logr.fit(x_train, y_train)
lr_end_time = current_milli_time()
models["Logistic Regression"]["duration"] = lr_end_time - lr_start_time
y_pred_lr = logr.predict(x_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
models["Logistic Regression"]["accuracy"] = calculate_AR(cm_lr)

# Random Forest Classification
rf_start_time = current_milli_time()
rfc = RandomForestClassifier(n_estimators=8, criterion='entropy')
rfc.fit(x_train, y_train)
rf_end_time = current_milli_time()
models["Random Forest Classification"]["duration"] = rf_end_time - rf_start_time
y_pred_rfc = rfc.predict(x_test)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
models["Random Forest Classification"]["accuracy"] = calculate_AR(cm_rfc)

# Decision Tree Classifier
dt_start_time = current_milli_time()
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)
dt_end_time = current_milli_time()
models["Decision Tree Classifier"]["duration"] = dt_end_time - dt_start_time
y_pred_dtc = dtc.predict(x_test)
cm_dtc = confusion_matrix(y_test, y_pred_dtc)
models["Decision Tree Classifier"]["accuracy"] = calculate_AR(cm_dtc)

# Naive Bayes Classifier
nb_start_time = current_milli_time()
gnb = GaussianNB()
gnb.fit(x_train, y_train)
nb_end_time = current_milli_time()
models["Naive Bayes Classifier"]["duration"] = nb_end_time - nb_start_time
y_pred_nb = gnb.predict(x_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
models["Naive Bayes Classifier"]["accuracy"] = calculate_AR(cm_nb)

# KNN Classifier
knn_start_time = current_milli_time()
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_end_time = current_milli_time()
models["KNN Classifier"]["duration"] = knn_end_time - knn_start_time
y_pred_knn = knn.predict(x_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
models["KNN Classifier"]["accuracy"] = calculate_AR(cm_knn)

# Deep Learning
dl_start_time = current_milli_time()
model = keras.Sequential([
    keras.Input(shape=(13)),
    layers.Dense(50, activation="relu"),
    layers.Dense(100, activation="relu"),
    layers.Dense(300, activation="relu"),
    layers.Dense(100, activation="relu"),
    layers.Dense(50, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="Nadam", loss="binary_crossentropy",
              metrics=['binary_accuracy'])
model.fit(x_train, y_train, epochs=60)
dl_end_time = current_milli_time()
models["Deep Learning"]["duration"] = dl_end_time - dl_start_time
y_pred_dl = model.predict(x_test)
y_pred_dl = (y_pred_dl > 0.7)
cm_dl = confusion_matrix(y_test, y_pred_dl)
models["Deep Learning"]["accuracy"] = calculate_AR(cm_dl)

# Find the model with the highest accuracy
best_model = max(models, key=lambda x: models[x]["accuracy"])
best_accuracy = models[best_model]["accuracy"]

# Printing the results
print(f"The model with the highest accuracy: {best_model}")
print(f"Accuracy rate: {best_accuracy}")
print("")

for model_name, model_data in models.items():
    print_values(model_name, model_data["accuracy"], model_data["duration"])
