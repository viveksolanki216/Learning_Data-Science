
# Date : 5 March 2022
# This File Creates a baseline model Using Random Forest, #
# And Compares over "Simple-Mean" Model (Super-Baseline Model), Where No Features Are Used. #
## Uses Log of target
## Uses GridSearch for Hyper-parameter Optimizations
## No Missing Data Handling
## No Outlier Handling
## Handle Multi-Collinearity
## No Feature Engineering
## No Model Selection
## No Feature Selection
# Results
## Performance : 0.1459 Error Score v/s 0.40 "Simple-Mean" Super Baseline Model

## Experiments:
    # Log Transformation on numeric variables did not improve much
    #

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

_home_dir, _in_dir, _out_dir = get_dir()
train_orig, test_orig, sample_subm = get_data()
train, test = train_orig.copy(deep=True), test_orig.copy(deep=True)
# Mean Baseline
print_numeric_super_baseline_model(train['SalePrice'], rmsle)

# --------------------------------
# Define Row Identifier and Target
Row_ID = 'Id'
target = 'SalePrice'
# Define Features

categorical_features = train.select_dtypes('object').columns.tolist()
numerical_features = [col for col in train.columns if col not in categorical_features if col not in [Row_ID, target]]
features = categorical_features+numerical_features

train_data_cat_enc, test_data_cat_enc, ohe_obj = onehotencode_categorical_vars(
    train[categorical_features],
    test[categorical_features]
)
# Create Final Feature Set and Targets for Train and Features for Test
X = pd.concat([train[numerical_features], train_data_cat_enc], axis=1)
X_test = pd.concat([test[numerical_features], test_data_cat_enc], axis=1)
y = train[target]
print(X.shape)
print(y.shape)
# Define Grid Parameters for GridSearchCV, where it will go and search over all parameter
grid = {
    'n_jobs': [8],
    'n_estimators': [100, 200,300,400,500],
    'max_features': ['sqrt', 'log2', 'auto'],
    'max_depth': [10, 15, 20, 25, 50],
    'min_samples_leaf': [5, 10, 30, 50],
    'max_samples': [.75, .85, 1],
    'random_state': [18]
}
rf_cv = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv=4,
                     scoring=make_scorer(rmsle))
rf_cv.fit(X, np.log(y + 1))
temp = pd.DataFrame(rf_cv.cv_results_)
# Make a Final Model from the best of all combinations of parameters in the Grid Search
best_parameters = {'max_depth': 10, 'max_features': 'auto',
                   'max_samples': 0.85, 'min_samples_leaf': 5,
                   'n_jobs': 8, 'oob_score': True, 'random_state':18,
                   'n_estimators': 600}
rf = RandomForestRegressor(**best_parameters)
rf.fit(X, np.log(y + 1))
# Out of Bag Error
rmsle(y, np.exp(rf.oob_prediction_) - 1)
feature_importance = get_feature_importance_random_forest(X.columns, rf.feature_importances_)
y_test_pred = rf.predict(X_test)
y_test_pred = np.exp(y_test_pred) - 1
#rmsle(y_test_pred, sample_subm['SalePrice_orig'])
sample_subm['SalePrice_orig'] = sample_subm['SalePrice']
sample_subm['SalePrice'] = y_test_pred
sample_subm[['Id', 'SalePrice']].to_csv(_out_dir + 'RF_V0.csv', index=False)
