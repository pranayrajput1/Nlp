import pandas as pd
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
import numpy as np

def encoding_categorical_features():
    train_data_df = pd.read_csv("SkillsDataSet.csv",na_filter=True, na_values='[]')
    train_data_df.dropna(subset=['Cleaned_skill_list'], inplace=True)
    department_column = train_data_df[['Deptartment']]
    encoder = LabelEncoder()
    train_data_df['Deptartment'] = encoder.fit_transform(department_column)
    return train_data_df

def splitting_dataset():
    train_data_df = encoding_categorical_features()
    requiredText = train_data_df['Cleaned_skill_list'].values.astype('U')
    requiredTarget = train_data_df['Deptartment'].values

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
    return X_train, X_test, y_train, y_test

def error_graph():
    X_train, X_test, y_train, y_test = splitting_dataset()
    error_rate = []

    for i in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 50), error_rate, color='blue', linestyle='solid', marker='o', markerfacecolor='green',
             markersize=10)
    plt.title('Error Rate v/s K value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.savefig("output.jpg")

def predict():
    X_train, X_test, y_train, y_test = splitting_dataset()
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7))
    trained_model = clf.fit(X_train, y_train)
    prediction = trained_model.predict(X_test)
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(trained_model.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(trained_model.score(X_test, y_test)))

    print("\n Classification report for classifier %s:\n%s\n" % (trained_model, metrics.classification_report(y_test, prediction)))


