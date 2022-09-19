import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn import metrics
import numpy as np

def pre_processing():
    data_df = pd.read_csv("SkillsDataSet.csv",na_filter=True, na_values='[]')
    data_df.dropna(subset=['Cleaned_skill_list'], inplace=True)
    data_df['Cleaned_skill_list'] = data_df.Cleaned_skill_list.apply(lambda x: get_pre_processed_value_column(x))
    test_df = data_df.groupby('Deptartment').tail(2)
    emails2remove = pd.merge(data_df, test_df, how='inner', on=['Cleaned_skill_list'])['Cleaned_skill_list']
    train_df = data_df[~data_df['Cleaned_skill_list'].isin(emails2remove)]
    return train_df

def get_pre_processed_value_column(text):
    text = text.lower().replace('\n', ' ')
    text = re.sub(' +', ' ', text)  # remove multiple spaces
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub('\s+ ', ' ', text)  # remove extra whitespace
    text = re.sub(r'\w*\d+\w*', '', text)  # Remove numbers
    text = text.strip()
    return text

def splitting_dataset():
    train_data_df = pre_processing()
    requiredText = train_data_df['Cleaned_skill_list'].values.astype('U')
    requiredTarget = train_data_df['Deptartment'].values
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    vectorizer = word_vectorizer.fit(requiredText)
    WordFeatures = vectorizer.transform(requiredText)
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
    return X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(13, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig("confusion_matrix.jpg")

def training_model_with_spilt_data():
    X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer = splitting_dataset()
    trained_model = KNeighborsClassifier()
    trained_model.fit(X_train, y_train)
    return trained_model

def training_model_with_full_data():
    X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer = splitting_dataset()
    trained_model = KNeighborsClassifier()
    trained_model.fit(WordFeatures, requiredTarget)
    return trained_model

def cross_validation():
    X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer = splitting_dataset()
    trained_model = training_model_with_spilt_data()
    cv_scores = cross_val_score(trained_model, WordFeatures, requiredTarget, cv=5)
    print("5 scores array:", cv_scores)
    print("cv_scores mean: {}".format(np.mean(cv_scores)))

def predict_on_splitData():
    X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer= splitting_dataset()
    trained_model = training_model_with_spilt_data()
    prediction = trained_model.predict(X_test)
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(trained_model.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(trained_model.score(X_test, y_test)))

    confusion_matrix_result = confusion_matrix(y_test, prediction)

    plot_confusion_matrix(cm=confusion_matrix_result,
                          normalize=False,
                          target_names=['Informatica_Lead', 'Devops', 'JAVA_Developer', 'AVP_Engineering','Data_Engineer', 'Rust', 'PM_&_Scrum_Master', 'GCP_Bigdata', 'SRE', 'Integration_Developer', 'QA', 'Scala_Developer', 'AIML_Engineer', 'Informatica_Onsite', 'Full_Stack_(Node + React)', 'Finance_Executive', 'Company_Secretary', 'Canada_Resumes', 'HR_Executive', 'Knoldus_Format_Resumes', 'AWS_Cloud_Engineer', 'Scrum_Master', 'GCP_Architect' 'Big_Data_Engineers', 'Calcite_Developer', 'Prod_Support', 'React_JS', 'C++_Rust', 'Node_JS', 'Data_Scientist'],
                          title="Confusion Matrix")

    print("\n Classification report for classifier %s:\n%s\n" % (trained_model, metrics.classification_report(y_test, prediction)))

def predict_on_testData(sample_test, num):
    X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer= splitting_dataset()
    trained_model = training_model_with_full_data()
    classes = trained_model.classes_
    word_feature_test = vectorizer.transform(sample_test)
    dense_prob_df = pd.DataFrame(trained_model.predict_proba(word_feature_test), columns=classes)
    transpose_df = dense_prob_df.T
    transpose_df['Deptartment'] = transpose_df.index
    transpose_df.reset_index(drop=True, inplace=True)
    transpose_df.rename(columns={0: 'Probabilty'}, inplace=True)
    prob = (transpose_df.nlargest(num, ['Probabilty'])).to_dict('records')
    return prob

