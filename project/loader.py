
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense
from keras import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from numpy import where
import matplotlib.pyplot as plt


def RandomForestModel(X_train, X_test, y_train, y_test):
    random_forest_model = RandomForestClassifier(n_estimators=10)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)
    bla = random_forest_model.score(X_test, y_test)
    print(random_forest_model.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    print("F1 score---- ", f1_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


def DecisionTreeModel(X_train, X_test, y_train, y_test):
    decision_tree_model = DecisionTreeClassifier(max_depth=14)
    decision_tree_model.fit(X_train, y_train)
    y_pred = decision_tree_model.predict(X_test)
    print(decision_tree_model.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    print("F1 score---- ", f1_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


def SupportVectorMachineModel(X_train, X_test, y_train, y_test):
    support_vector_machine_model = SVC()
    support_vector_machine_model.fit(X_train, y_train)
    y_pred = support_vector_machine_model.predict(X_test)
    print(support_vector_machine_model.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    print("F1 score---- ", f1_score(y_test, y_pred))
    # print(naive_bayes_model.score(X_test, y_test))
    print(accuracy_score(y_test, y_pred))


def NaiveBayesModel(X_train, X_test, y_train, y_test):
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, y_train)
    y_pred = naive_bayes_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("F1 score---- ", f1_score(y_test, y_pred))
    #print(naive_bayes_model.score(X_test, y_test))
    print(accuracy_score(y_test, y_pred))


def CNNModel(X_train, X_test, y_train, y_test):
    epochs = 20
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=32, kernel_size=2, activation="relu", input_shape=X_train[0].shape))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Conv1D(64, 2, activation="relu"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Dense(1, activation='sigmoid'))
    cnn_model.summary()
    print(cnn_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1))
    # accuracy_score(y_test, y_test1)


def SMOTE_data(data):
    X = data.drop(columns=['Class']).values
    y = data['Class'].values
    Counter(y)
    # transform the dataset
    smt = SMOTE(random_state=0)
    X, y = smt.fit_resample(X, y)
    return X, y


# klasican undersampling ulaznog dataseta -> smanjujemo broj validnih transakcija na broj nevalidnih
def undersampling(loaded_data):
    non_fraud = loaded_data[loaded_data['Class'] == 0]
    fraud = loaded_data[loaded_data['Class'] == 1]
    num_of_fraud = fraud.shape[0]
    non_fraud.sample(n=num_of_fraud)
    data2 = non_fraud.sample(n=num_of_fraud)
    return pd.concat([fraud, data2], axis=0)


# klasican oversampling ulaznog dataseta -> povecavamo broj nevalidnih transakcija na broj validnih
def oversampling(loaded_data):
    max_size = loaded_data['Class'].value_counts().max()
    lst = [loaded_data]
    for class_index, group in loaded_data.groupby('Class'):
        lst.append(group.sample(max_size - len(group), replace=True))
    frame_new = pd.concat(lst)
    return frame_new


def creating_training_and_test_set(X_scaler, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=2,
                                                        stratify=y)  # 80:20 = training: test
    return X_train, X_test, y_train, y_test


def preprocessing(data):
    # print(data.info()) #prikaz informacija o datom skupu podataka (da li ima null vrednosti i kog su tipa)
    # print(data.isnull().values.any()) #provera da li u celom skupu ima null vrednosti

    # preparation of data
    time_delta = pd.to_timedelta(data['Time'], unit='s')
    data['Day_time'] = time_delta.dt.components.days.astype(int)
    data['Hour'] = time_delta.dt.components.hours.astype(int)
    data['Minute'] = time_delta.dt.components.minutes.astype(int)

    data.drop('Time', axis=1, inplace=True)
    data.drop('Day_time', axis=1, inplace=True)
    data.drop('Minute', axis=1, inplace=True)

    new_Data = undersampling(data)

    # splitting data
    X = new_Data.drop('Class', axis=1)
    y = new_Data['Class']

    # scaling
    standard_scaler = StandardScaler()
    x_scaler = standard_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = creating_training_and_test_set(x_scaler, y)
    print("\nNaive Bayes undersampling ----------------------------------")
    NaiveBayesModel(X_train, X_test, y_train, y_test)


def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='midnightblue')
        ax.set_title(feature + " Distribution", color='DarkRed')
        ax.set_yscale('log')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    loaded_data = pd.read_csv('creditcard.csv', sep=',')  # ucitan skup podataka
    # print(loaded_data['Class'].value_counts())
    # draw_histograms(loaded_data, loaded_data.columns, 8, 4)

    preprocessing(loaded_data)

    pass



