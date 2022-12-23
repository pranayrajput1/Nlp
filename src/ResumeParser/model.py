import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn import metrics
import numpy as np

def pre_processing():
    """
    This function is used to perform pre-processing on the dataset
    """
    data_df = pd.read_csv("New_Skill_Set.csv",na_filter=True, na_values='[]')
    data_df.dropna(subset=['Cleaned_skill_list'], inplace=True)
    data_df['Cleaned_skill_list'] = data_df.Cleaned_skill_list.apply(lambda x: get_pre_processed_value_column(x))
    test_df = data_df.groupby('Deptartment').tail(2)
    emails2remove = pd.merge(data_df, test_df, how='inner', on=['Cleaned_skill_list'])['Cleaned_skill_list']
    train_df = data_df[~data_df['Cleaned_skill_list'].isin(emails2remove)]
    test_df.to_csv("test_df.csv", index=False)
    train_df.to_csv("train_df.csv", index=False)
    return train_df

def get_pre_processed_value_column(text):
    """
    This function is used to perform pre-processing on the skills column
    @input text : skills
    @input dtype : string
    @return text : pre-processed skills string
    @return dtype : string
    """
    text = text.lower().replace('\n', ' ')
    text = re.sub(' +', ' ', text)  # remove multiple spaces
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub('\s+ ', ' ', text)  # remove extra whitespace
    text = re.sub(r'\w*\d+\w*', '', text)  # Remove numbers
    text = text.strip()
    return text

def splitting_dataset():
    """
    This function is used to perform spilting of the dataset and applying word vectorizer and transformation to the skills column
    """
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
    print("\n Classification report for classifier %s:\n%s\n" % (trained_model, metrics.classification_report(y_test, prediction)))

def similarity(list1, list2):
    element_match = set(list1[0]) & set(list2)
    element_match_count = len(element_match)
    primary_list_count = len(list2)
    score = (element_match_count / primary_list_count)
    return score

def similarity_for_csv(list1, list2):
    element_match = set(list1) & set(list2)
    element_match_count = len(element_match)
    primary_list_count = len(list2)
    score = (element_match_count / primary_list_count)
    return score

def predict_on_one_testData(sample_test):
    X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer= splitting_dataset()
    trained_model = training_model_with_full_data()
    classes = trained_model.classes_
    word_feature_test = vectorizer.transform([sample_test])
    dense_prob_df = pd.DataFrame(trained_model.predict_proba(word_feature_test), columns=classes)
    copy_dense_prob_df = dense_prob_df.copy()
    skill_lst = sample_test.split()
    copy_dense_prob_df['Resume_Skills'] = pd.Series([skill_lst])

    # constructing primary skills list
    primary_skills_dict = {'Devops' : ['cicd', 'linux', 'cloud'], 'Java': ['java', 'spring'], 'Scala' : ['scala', 'sbt', 'spark'],
                           'ML Engineer' : ['machine', 'learning','python'], 'Finance': ['gaap', 'excel', 'accounting'],
                           'Frontend' : ['javascript', 'html', 'css'], 'Project Manager' : ['agile', 'management', 'scrum']}

    # calculating scores by primary skills here
    primary_skills_score_lst = []
    primary_skills_department_lst = []
    primary_skills_lst = []
    for key, values in primary_skills_dict.items():
        score = similarity(copy_dense_prob_df['Resume_Skills'].tolist(), values)
        primary_skills_score_lst.append(score)
        primary_skills_department_lst.append(key)
        primary_skills_lst.append(values)

    copy_dense_prob_df.drop(['Resume_Skills'], axis=1, inplace=True)
    copy_dense_prob_df = copy_dense_prob_df.append(pd.DataFrame([primary_skills_score_lst],columns=primary_skills_department_lst),ignore_index=True)
    copy_dense_prob_df = copy_dense_prob_df.append(pd.DataFrame([primary_skills_lst], columns=primary_skills_department_lst), ignore_index=True)
    transpose_df = copy_dense_prob_df.T
    transpose_df['Deptartment'] = transpose_df.index
    transpose_df['Secondary Skills Scores'] = transpose_df[0]
    transpose_df['Primary Skills Scores'] = transpose_df[1]
    transpose_df['Resume Skills'] = sample_test
    transpose_df['Primary Skills'] = transpose_df[2]
    transpose_df.drop([0,1,2], axis=1, inplace=True)
    transpose_df = transpose_df[['Deptartment', 'Resume Skills', 'Primary Skills', 'Secondary Skills Scores', 'Primary Skills Scores']]
    transpose_df.to_csv("updated_single_new_results.csv",index=False)
    return transpose_df

def predict_on_csv_testData(sample_test):
    X_train, X_test, y_train, y_test, WordFeatures, requiredTarget, vectorizer = splitting_dataset()
    trained_model = training_model_with_full_data()
    classes = trained_model.classes_
    requiredText = sample_test['Cleaned_skill_list'].values.astype('U')
    word_feature_test = vectorizer.transform(requiredText)
    dense_prob_df = pd.DataFrame(trained_model.predict_proba(word_feature_test), columns=classes)
    copy_dense_prob_df = dense_prob_df.copy()
    comma_separated_skill_lst = []
    for i in requiredText:
        skill_lst = i.split()
        comma_separated_skill_lst.append((skill_lst))

    copy_dense_prob_df['Resume_Skills'] = comma_separated_skill_lst
    # constructing primary skills list
    primary_skills_dict = {'Devops' : ['cicd', 'linux', 'cloud'], 'Java': ['java', 'spring'], 'Scala' : ['scala', 'sbt', 'spark'],
                           'ML Engineer' : ['machine', 'learning','python'], 'Finance': ['gaap', 'excel', 'accounting'],
                           'Frontend' : ['javascript', 'html', 'css'], 'Project Manager' : ['agile', 'management', 'scrum']}

    # calculating scores by primary skills here
    score_lst = []
    deptartment_lst = []
    pskill_lst = []
    resume_skill = []
    for i in copy_dense_prob_df['Resume_Skills']:
        primary_skills_score_lst = []
        primary_skills_department_lst = []
        primary_skills_lst = []
        for key, values in primary_skills_dict.items():
            score = similarity_for_csv(i, values)
            primary_skills_score_lst.append(score)
            primary_skills_department_lst.append(key)
            primary_skills_lst.append(values)
        score_lst.append(primary_skills_score_lst)
        deptartment_lst.append(primary_skills_department_lst)
        pskill_lst.append(primary_skills_lst)
        resume_skill.append(i)

    copy_dense_prob_df['new'] = score_lst
    copy_dense_prob_df['Primary Skills'] = pskill_lst
    copy_dense_prob_df[['Devops_Primary_Score', 'Java_Primary_Score', 'Scala_Primary_Score', 'ML_Enginner_Primary_Score', 'Finance_Primary_Score', 'Frontend_Primary_Score', 'Project_Manager_Primary_Score']] = pd.DataFrame(copy_dense_prob_df.new.tolist(), index=copy_dense_prob_df.index)
    copy_dense_prob_df = copy_dense_prob_df[['Resume_Skills','Primary Skills','Devops','Devops_Primary_Score', 'Java', 'Java_Primary_Score','Scala', 'Scala_Primary_Score', 'ML Engineer','ML_Enginner_Primary_Score', 'Finance', 'Finance_Primary_Score', 'Frontend', 'Frontend_Primary_Score', 'Project Manager', 'Project_Manager_Primary_Score']]
    copy_dense_prob_df.to_csv("new_data_result.csv", index=False)
    return copy_dense_prob_df



