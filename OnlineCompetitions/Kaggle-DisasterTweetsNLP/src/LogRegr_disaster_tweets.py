import os

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Get all the directories path

home_dir, in_dir, out_dir, plots_dir, analyses_dir = get_dirs(os.getcwd())
# Read all the data
train, test, sample_sub = get_data(in_dir)

train['clean_text'] = preprocess_text(train['text'])[0]
test['clean_text'] = preprocess_text(test['text'])[0]
doc_list_universe = pd.concat([train['clean_text'], test['clean_text']])
# Converting texts to BoWs with tfidf scoring.
tfidf = TfidfVectorizer(binary=True, max_df=0.4)
tdm_universe = tfidf.fit_transform(doc_list_universe)
tdm1 = tfidf.transform(train['clean_text'])
tdm2 = tfidf.transform(test['clean_text'])

svd = TruncatedSVD(n_components=3000, n_iter=2, random_state=42)
svd.fit(tdm_universe)
tdm_lsi1 = svd.transform(tdm1)
tdm_lsi2 = svd.transform(tdm2)
svd.explained_variance_ratio_.sum()
svd.explained_variance_.sum()

estimator = LogisticRegression()
parameters = {'penalty': ['l2'],
              'C': [0.001, 0.1, 3],
              'class_weight': ['balanced'],
              'solver': ['liblinear']}
grid_log = GridSearchCV(estimator, parameters, n_jobs=-1, scoring='f1', cv=5, return_train_score=True)
grid_log.fit(tdm_lsi1, train['target'].values)
grid_results = pd.DataFrame(grid_log.cv_results_)

from sklearn.ensemble import GradientBoostingClassifier

estimator = GradientBoostingClassifier()
parameters = {'n_estimators': [200],
              'subsample': [1],
              'min_samples_split': [20],
              'min_samples_leaf': [10],
              'max_features': [None]}
grid = GridSearchCV(estimator, parameters, n_jobs=-1, scoring='f1', cv=4, return_train_score=True)
grid.fit(tdm_lsi1, train['target'].values)
grid_results2 = pd.DataFrame(grid.cv_results_)
grid.best_params_

estimator = GradientBoostingClassifier(
    **{'max_features': None, 'min_samples_leaf': 10, 'min_samples_split': 20, 'n_estimators': 200, 'subsample': 1})
estimator.fit(tdm_lsi1, train['target'].values)
y_pred = estimator.predict(tdm_lsi2)

sample_sub['target'] = y_pred

sample_sub.to_csv(f'{out_dir}LSI_GBR.csv', index=False)
