
# Date : 6 March 2022
# This File Creates a model above baseline model (Random Forest), #
## Uses Log of target
## Uses GridSearch for Hyper-parameter Optimizations
## Handling Missing Data
## Doing Feature Engineering
    # Log Transformation of numeric variables has substantial impace on score.
## No Outlier Handling
## No Handle Multi-Collinearity
## No Model Selection
## No Feature Selection
# Results
## Performance : 0.1459 Error Score v/s 0.40 "Simple-Mean" Super Baseline Model

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge, HuberRegressor
from sklearn.cross_decomposition import PLSRegression

_home_dir, _in_dir, _out_dir = get_dir()
train_orig, test_orig, sample_subm = get_data()
train, test = train_orig.copy(deep=True), test_orig.copy(deep=True)
# Mean Baseline
print_numeric_super_baseline_model(train['SalePrice'], rmsle)

missings_cols_before = check_missing_rate_train_test(train, test)
train, test = impute_missings(train, test)
missings_cols = check_missing_rate_train_test(train, test)

# Feature Engineering
train = feature_engineering(train)
test = feature_engineering(test)
train = train[ ~train['Id'].isin([692,1183])].reset_index(drop=True)

# Find Columns that are numeric, has lot of values, and needs log transformation
numeric_cols_to_transform=[col for col in train.columns if train[col].nunique() >=120 if col not in ['Id', 'SalePrice']]
# --------------------------------
# Define Row Identifier and Target
Row_ID = 'Id'
target = 'SalePrice'
# Define Features

categorical_features = train.select_dtypes('object').columns.tolist()
numerical_features = [col for col in train.columns if col not in categorical_features if col not in [Row_ID, target]]
features = categorical_features+numerical_features

# Convert Categorical variables to dummy numeric variables
train_data_cat_enc, test_data_cat_enc, ohe_obj = onehotencode_categorical_vars(
    train[categorical_features],
    test[categorical_features]
)
# Transform Numeric variables to log, polynomial features
X_num = train[numeric_cols_to_transform].copy(deep=True)
X_test_num = test[numeric_cols_to_transform].copy(deep=True)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
poly.fit(X_num)
poly.powers_
X_num_poly=poly.transform(X_num)
X_test_num_poly=poly.transform(X_test_num)

for col in numeric_cols_to_transform:
    print(col)
    print(train[col].describe())
    train[col] = np.log(train[col] + 3)
    test[col] = np.log(test[col] + 3)
    print(train[col].describe())

# Create Final Feature Set and Targets for Train and Features for Test
X = np.concatenate([train_data_cat_enc.values,
                    train[numerical_features].values,
                    X_num_poly], axis=1)
X_test = np.concatenate([test_data_cat_enc.values,
                         test[numerical_features].values,
                         X_test_num_poly], axis=1)
y = train[target]
print(X.shape)
print(y.shape)
# Define Grid Parameters for GridSearchCV, where it will go and search over all parameter


models_grid=[
    {
        'name': 'ridge regression',
        'estimator': Ridge(),
        'hyperparameters': {
            'alpha': np.arange(2, 50, 1),  # [0.5, 1, 2.4, 5, 7.5, 10, 20, 50],
            'normalize': [True]
        }
    }
#{ Huber Regression is not working maybe beacuse of 0/1 variables
 #       'name': 'huber regression',
 #       'estimator': HuberRegressor(),
 #       'hyperparameters': {
 #           'max_iter': [100],
 #           'alpha': [0.001, 0.005, 0.01, 0.05, 0.5, 1],
 #           'epsilon': [1.1, 1.35]
 #       }
  #  }
]

all_models_score = pd.DataFrame()
for model in models_grid:
    print(model['name'])
    gs=GridSearchCV(model['estimator'], param_grid=model['hyperparameters'],
                    cv=2, n_jobs=-1, scoring='neg_root_mean_squared_error')
    gs.fit(X, np.log(y + 1))
    models_score = pd.DataFrame(gs.cv_results_)
    models_score['estimator'] = model['name']
    all_models_score = pd.concat([all_models_score, models_score], axis=0)
    print('best score: ', gs.best_score_)
    print('best parameters ; ', gs.best_params_)
    print('best model: ', gs.best_estimator_)
    print('---------------------------------\n')


# A Final best model
elastic_model =  ElasticNet(alpha=0.0017, l1_ratio=0.35, max_iter=10000)
elastic_model.fit(X, np.log(y + 1))
y_test_pred = elastic_model.predict(X_test)
y_test_pred = np.exp(y_test_pred) - 1
#rmsle(y_test_pred, sample_subm['SalePrice_orig'])
#sample_subm['SalePrice_orig'] = sample_subm['SalePrice']
sample_subm['SalePrice'] = y_test_pred
sample_subm[['Id', 'SalePrice']].to_csv(_out_dir + 'ElasticNet_V4.csv', index=False)

coefs = pd.DataFrame({'features':X.columns, 'coefs': elastic_model.coef_})

train_orig['SalePrice_pred_ElasticNet'] = np.exp(elastic_model.predict(X))-1
train_orig.to_csv(_home_dir+'analysis/train_elastic_net.csv', index=False)

sns.scatterplot(train['SalePrice'], train['SalePrice_pred'])
sns.scatterplot(train['SalePrice'], train['SalePrice'])
plt.savefig(_home_dir+'plots/ElasticNet_Preds.png')
plt.show()
