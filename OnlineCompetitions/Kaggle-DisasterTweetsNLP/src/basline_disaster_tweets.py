import os

# Get all the directories path
import numpy as np
import pandas as pd

home_dir, in_dir, out_dir, plots_dir, analyses_dir = get_dirs(os.getcwd())
# Read all the data
train, test, sample_sub = get_data(in_dir)

train['clean_tweet'] = preprocess_text(train['text'])[0]
test['clean_tweet'] = preprocess_text(test['text'])[0]

# create 5 samples of train dataset for cross-validation
train['k'] = np.random.randint(low=1, high=6, size=train.shape[0])
train['k'].value_counts(normalize=True)

knn_k = 10
all_scores = []
for i in train['k'].unique():
    print(i)
    train_set = train[train.k != i].reset_index(drop=True)
    valid_set = train[train.k == i].reset_index(drop=True)

    keyword_scores = (valid_set['keyword'].values == train_set['keyword'].values[:, np.newaxis]).astype(int)
    text_score = get_similarity_score_LSI(train_set['clean_tweet'],
                                          valid_set['clean_tweet'],
                                          pd.concat([train_set['clean_tweet'], valid_set['clean_tweet']], axis=0))
    similarity = text_score + keyword_scores
    sorted_indices = np.argsort(similarity,
                                axis=0)  # Sort the matrix along cols, and keep indices in sorted order on values
    top_indices = sorted_indices[:-1 * (knn_k + 1):-1, :]  # fetches values(indices) from last to last-k position
    top_scores = similarity[top_indices, np.arange(similarity.shape[1])]  # Fancy indexing, first index as column vector
    top_targets = train_set['target'].values[top_indices].astype('float')
    valid_set['target_preds'] = np.mean(top_targets, axis=0)
    print('missing values', valid_set['target_preds'].isnull().sum())
    # valid_set['target_preds'].fillna(0, inplace=True)
    valid_set['target_preds'] = (valid_set['target_preds'] > 0.4).astype('int')
    print(valid_set['target_preds'].value_counts(normalize=True))
    print(get_f1_score(valid_set['target'], valid_set['target_preds']))
    f1_score, _, _ = get_f1_score(valid_set['target'], valid_set['target_preds'])
    all_scores = all_scores + [f1_score]

print('Final score : ', np.array(all_scores).mean())

print(i)
train_set = train
valid_set = test

keyword_scores = (valid_set['keyword'].values == train_set['keyword'].values[:, np.newaxis]).astype(int)
text_score = get_similarity_score_LSI(train_set['clean_tweet'],
                                      valid_set['clean_tweet'],
                                      pd.concat([train_set['clean_tweet'], valid_set['clean_tweet']], axis=0))
similarity = text_score + keyword_scores
sorted_indices = np.argsort(similarity,
                            axis=0)  # Sort the matrix along cols, and keep indices in sorted order on values
top_indices = sorted_indices[:-1 * (knn_k + 1):-1, :]  # fetches values(indices) from last to last-k position
top_scores = similarity[top_indices, np.arange(similarity.shape[1])]  # Fancy indexing, first index as column vector
top_targets = train_set['target'].values[top_indices].astype('float')
valid_set['target_preds'] = np.mean(top_targets, axis=0)
print('missing values', valid_set['target_preds'].isnull().sum())
# valid_set['target_preds'].fillna(0, inplace=True)
valid_set['target_preds'] = (valid_set['target_preds'] > 0.4).astype('int')
print(valid_set['target_preds'].value_counts(normalize=True))

sample_sub['target'] = valid_set['target_preds']
sample_sub.to_csv(f'{out_dir}LSI_KNN10.csv', index=False)
