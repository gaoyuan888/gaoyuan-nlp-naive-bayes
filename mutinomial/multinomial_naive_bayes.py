from sklearn import datasets

news = datasets.fetch_20newsgroups(subset='all')
print(news.keys())

split_rate = 0.8
split_size = int(len(news.data) * split_rate)
X_train = news.data[:split_size]
y_train = news.target[:split_size]
X_test = news.data[split_size:]
y_test = news.target[split_size:]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Tokenizing text
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
# Tf
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
# Tf_idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB

# create classifier
clf = MultinomialNB().fit(X_train_tfidf, y_train)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# using classifier to predict
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, news.target_names[category]))

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, \
    CountVectorizer  # nbc means naive bayes classifier

nbc_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])
nbc_2 = Pipeline([
    ('vect', HashingVectorizer(non_negative=True)),
    ('clf', MultinomialNB()),
])
nbc_3 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])
# classifier
nbcs = [nbc_1, nbc_2, nbc_3]

from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from scipy.stats import sem
import numpy as np


# cross validation function
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k folds
    cv = KFold(K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


for nbc in nbcs:
    evaluate_cross_validation(nbc, X_train, y_train, 10)

nbc_4 = Pipeline([
    ('vect', TfidfVectorizer(
        token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b", )
     ),
    ('clf', MultinomialNB()),
])
evaluate_cross_validation(nbc_4, X_train, y_train, 10)


import nltk
# nltk.download()
stopwords = nltk.corpus.stopwords.words('english')
nbc_5 = Pipeline([
   ('vect', TfidfVectorizer(
               stop_words=stopwords,
               token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
   )),
   ('clf', MultinomialNB()),
])
evaluate_cross_validation(nbc_5, X_train, y_train, 10)

nbc_6 = Pipeline([
    ('vect', TfidfVectorizer(
                stop_words=stopwords,
                token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB(alpha=0.01)),
])
evaluate_cross_validation(nbc_6, X_train, y_train, 10)


pipeline = Pipeline([
('vect',CountVectorizer()),
('tfidf',TfidfTransformer()),
('clf',MultinomialNB()),
]);
parameters = {
    'vect__max_df': (0.5, 0.75),
    'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)

from time import time
t0 = time()
grid_search.fit(X_train, y_train)
print "done in %0.3fs" % (time() - t0)
print "Best score: %0.3f" % grid_search.best_score_


from sklearn import metrics
best_parameters = dict()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print "\t%s: %r" % (param_name, best_parameters[param_name])
pipeline.set_params(clf__alpha = 1e-05,
                    tfidf__use_idf = True,
                    vect__max_df = 0.5,
                    vect__max_features = None)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print np.mean(pred == y_test)


from sklearn import metrics
import numpy as np
#print X_test[0], y_test[0]
for i in range(20):
    print str(i) + ": " + news.target_names[i]
predicted = pipeline.fit(X_train, y_train).predict(X_test)
print np.mean(predicted == y_test)
print metrics.classification_report(y_test, predicted)