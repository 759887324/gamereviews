import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline



def read_file():
    data = pd.read_csv('data/data_pre1.csv', 'rb', engine='python')
    data = data.sample(frac=1, random_state=2020)
    data = list(data['comment,label_1,label_2,label_3'])

    label = []
    contents = []
    for i in data:
        a = i.split(',')

        if a[2] == '':
            label.append(a[1:2])
        elif a[3] == '':
            label.append(a[1:3])
        else:
            label.append(a[1:])
        contents.append(a[0])

    return contents, label


def read_category():
    with open('\data\cail_vocab\label.txt') as fp:
        categories = [_.strip() for _ in fp.readlines()]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def process_file(cat_to_id):
    contents, labels = read_file()

    data_id, label_id = [], []
    for i in range(len(contents)):
        label_id.append([cat_to_id[x] for x in labels[i]])

    class_matrix = np.eye(len(cat_to_id))
    y_pad = np.array(list(map(lambda x: np.sum(class_matrix[x], axis=0), label_id)))
    x_pad_train = contents[0:7000]
    x_pad_test = contents[7000:]

    y_pad_train = y_pad[0:7000]
    y_pad_test = y_pad[7000:]
    return x_pad_train, y_pad_train, x_pad_test, y_pad_test



categories, cat_to_id = read_category()
X_train,y_train,X_test,y_test=process_file(cat_to_id)


#cv = CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')
#feature = cv.fit_transform(X_train)
#print(feature.shape)
#print()
#print(feature)


tfidf = TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')
feature = tfidf.fit_transform(X_train)
#print(feature.shape)
#print()
#print(feature)



from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import classification_report

def print_evaluation_scores(y_test, predicted):
    accuracy=accuracy_score(y_test, predicted)
    hamming_score=hamming_loss(y_test,predicted)
    average=average_precision_score(y_test,predicted)

    print("accuracy:",accuracy)
    print('precision',average)
    print('hamming loss',hamming_score)
    print(classification_report(y_test,predicted))

'''
NB_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01))),
])

NB_pipeline.fit(X_train, y_train)
predicted = NB_pipeline.predict(X_test)
print_evaluation_scores(y_test, predicted)
'''
SVC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(SVC(gamma=0.2), n_jobs=1)),
])

SVC_pipeline.fit(X_train, y_train)
predicted = SVC_pipeline.predict(X_test)
print_evaluation_scores(y_test, predicted)
'''
LogReg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(LogisticRegression(C=0.7), n_jobs=1)),
])  

LogReg_pipeline.fit(X_train, y_train)
predicted = LogReg_pipeline.predict(X_test)
print_evaluation_scores(y_test, predicted)
'''





