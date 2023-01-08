
# Date : 6 March 2022
# This File Creates a model above baseline model (Random Forest), #
## Uses Log of target
## Uses GridSearch for Hyper-parameter Optimizations
## Handling Missing Data
## Doing Feature Engineering
    # Log Transformation of numeric variables has substantial impace on score.
## Model Selection (GridSearchCV)
## Feature Selection (Lasso)
## No Outlier Handling
## No Handle Multi-Collinearity

# Results
## Performance : 0.12463 Error Score For ElasticNet (alpha=0.0017, l1_ratio=0.35, max_iter=10000)
    ## Log Transformation on Numeric features that has distinct values more then 100, impacts scores a lot
    ## That means linear models does much depends on numeric feature having normal distribution
    ## 0.12463 v/s 0.13156 before log transformation
    ## 0.12409 v/s 0.12463, very slight improvement using the boxcox instead log.


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
from scipy.stats import skew
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
from scipy.special import inv_boxcox1p
from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,PowerTransformer

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
numeric_features = train.dtypes[train.dtypes != "object"].index.tolist()
numeric_features = list(set(numeric_features).difference(['Id', 'SalePrice']))
#num_cols = [col for col in train.columns if train[col].nunique() >=120 if col not in ['Id', 'SalePrice']]
skewed_features = get_high_skewed_features(train, numeric_features, 0.9)
boxcox_param = [boxcox_normmax(train[col]+2) for col in skewed_features]

distri_before_transform = train[skewed_features].describe()
for feature, power in zip(skewed_features, boxcox_param):
    train[feature] = boxcox1p(train[feature]+1, power)
    test[feature] = boxcox1p(test[feature] + 1, power)
distri_after_transform = train[skewed_features].describe()
skewed_features = get_high_skewed_features(train, skewed_features, 0.9)


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

#std_scaler = StandardScaler()
#X = std_scaler.fit_transform(X)
#X_test = std_scaler.fit_transform(X_test)
#y


# First Linear Model Grid
models_grid=[
    {
        'name': 'linear regression',
        'estimator':LinearRegression(),
        'hyperparameters':{
            'normalize':[True, False]
        }
    },
    {
        'name': 'ElasticNet',
        'estimator': ElasticNet(),
        'hyperparameters':{
            'max_iter':[10000],
            'alpha': np.arange(0.0001, 0.005, 0.0002), # Initial parameters [0.0001,  0.001, 0.005, 0.01, 0.05, 0.5, 1, 2],
            'l1_ratio': np.arange(0.1, 0.5, 0.05), #[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            'normalize': [True, False]
        }
    },
    {
        'name': 'ridge regression',
        'estimator': Ridge(),
        'hyperparameters': {
            'alpha': np.arange(5, 10, 0.25), #[0.5, 1, 2.4, 5, 7.5, 10, 20, 50],
            'normalize': [True, False]
        }
    },
    {
        'name': 'lasso regression',
        'estimator': Lasso(),
        'hyperparameters': {
            'max_iter': [10000],
            'alpha': np.arange(0.0001, 0.005, 0.0001),# Starting Params : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            'normalize': [True, False]
        }
    },
    {
        'name': 'PLSRegression',
        'estimator': PLSRegression(),
        'hyperparameters': {
            'max_iter': [1000],
            'n_components': np.arange(1, 100, 1),
            'scale': [True, False]
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

from sklearn.linear_model import SGDRegressor
# Some other linear models
# Seems like SGD Regressor is not able to converge
models_grid =[
    {
        'name' : 'Stochastic Gradient Regressor',
        'estimator' : SGDRegressor(),
        'hyperparameters' : {
            #'loss' : ['squared_loss'],
            'penalty' : ['elasticnet'],
            'alpha': np.arange(0.0001, 0.005, 0.0002),
            'l1_ratio': np.arange(0.1, 0.5, 0.05),  # [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            #'epsilon' : np.arange(0.001, 2, 0.2),
            'max_iter': [10000],
            'eta0' :[5e-1], 'tol':[1e-8]
        }
    }
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
elastic_model =  ElasticNet(alpha=0.0021, l1_ratio=0.25, max_iter=10000)
elastic_model.fit(X, np.log(y + 1))
y_test_pred = elastic_model.predict(X_test)
y_test_pred = np.exp(y_test_pred) - 1
#rmsle(y_test_pred, sample_subm['SalePrice_orig'])
#sample_subm['SalePrice_orig'] = sample_subm['SalePrice']
sample_subm['SalePrice2'] = y_test_pred
sample_subm[['Id', 'SalePrice']].to_csv(_out_dir + 'ElasticNet_V4_Expr_boxcox_all_num.csv', index=False)

coefs = pd.DataFrame({'features':X.columns, 'coefs': elastic_model.coef_})

train_orig['SalePrice_pred_ElasticNet'] = np.exp(elastic_model.predict(X))-1
train_orig.to_csv(_home_dir+'analysis/train_elastic_net.csv', index=False)

sns.scatterplot(train['SalePrice'], train['SalePrice_pred'])
sns.scatterplot(train['SalePrice'], train['SalePrice'])
plt.savefig(_home_dir+'plots/ElasticNet_Preds.png')
plt.show()
