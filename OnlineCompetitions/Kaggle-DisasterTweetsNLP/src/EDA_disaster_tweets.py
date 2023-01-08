import os

# Get all the directories path
home_dir, in_dir, out_dir, plots_dir, analyses_dir = get_dirs(os.getcwd())
# Read all the data
train, test, sample_sub = get_data(in_dir)

print(train.shape)
print(test.shape)
print(train.info())
print(test.info())
print(train.describe())

# target variable
print(train.target.value_counts())
# 43% times tweet was related to target
print(train.target.value_counts(normalize=True))
# Apart from target, there is keyword and location related to that tweet is provided.
# for less than 1% keyword is missing.
print(train.keyword.isnull().sum() / train.shape[0] * 100)
print(test.keyword.isnull().sum() / test.shape[0] * 100)
keywords = pd.concat([
    train.keyword.value_counts(normalize=True),
    test.keyword.value_counts(normalize=True)
], axis=1, join='outer')  # By default, its outer, so no need to specify join

# Things to keep in mind is that, there are a lot of unique location and not well-structured.
# May need a heavy cleaning.
# Location is missing 33% times.
print(train.location.isnull().sum() / train.shape[0] * 100)
print(test.location.isnull().sum() / test.shape[0] * 100)
location = pd.concat([
    train.location.value_counts(normalize=True),
    test.location.value_counts(normalize=True)
], axis=1, join='outer')  # By default its outer, so no need to specify join
