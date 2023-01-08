
# Date : 10 March 2022
# This File Creates a model above baseline model (Random Forest), #
## Uses Log of target
## Uses GridSearch for Hyper-parameter Optimizations
## Handling Missing Data
## Doing Feature Engineering
    # Log Transformation of numeric variables has substantial impace on score.
## Model Selection (GridSearchCV)
## Feature Selection (Lasso)
## Meta Model
## No Outlier Handling
## No Handle Multi-Collinearity

# Results
## Exp1 Performance : 0.12409 Error Score For ElasticNet (alpha=0.0017, l1_ratio=0.35, max_iter=10000)
    ## Log Transformation BoxCox
## Exp2 Performance : Combining Several Models
    ## Just taking average of ElasticNet & RGBM gives us .12237 (v/s 0.12409 on elsticnet) even though RGBM performing
        ## around 0.13
    ## Now using a Voting classifier instead of avg, and gives us .12213 around.
    ## Using Stacking Regressor gives 0.12131
    # Tried adding other models in the stacking regression, but Elastic+RGBM with LineaRegression seem the best model


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    train[feature] = boxcox1p(train[feature] + 1, power)
    test[feature] = boxcox1p(test[feature] + 1, power)
distri_after_transform = train[skewed_features].describe()
skewed_features = get_high_skewed_features(train, numeric_features, 0.9)


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

models_grid=[
    {
        'name': 'ElasticNet',
        'estimator': ElasticNet(),
        'hyperparameters': {
            'max_iter': [10000],
            'alpha': [0.0001,  0.001, 0.005, 0.01, 0.05, 0.5, 1, 2],#np.arange(0.0001, 0.005, 0.0002),
            # Initial parameters ,
            'l1_ratio':  [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9], #np.arange(0.1, 0.5, 0.05),  #
            'normalize': [True, False]
        }
    },
        {
            'name': 'GradientBoostingRegressor',
            'estimator':GradientBoostingRegressor(),
            'hyperparameters':{
                #'loss': ['huber_loss'],
                'n_estimators': [100, 350,400,450, 500],
                'max_features': ['log2', 'sqrt', 'auto'],
                'max_depth': [15, 20,25, 30],
                'min_samples_leaf': [ 20, 25, 30, 35, 50],
                'subsample': [.5, .75, .85, 1],
                #'criterion' :['squared_error'],
                'random_state': [18]
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

elastic_model =  ElasticNet(alpha=0.0021, l1_ratio=0.25, max_iter=10000)
gbr_model =   GradientBoostingRegressor(max_depth=20, max_features='log2',
                          min_samples_leaf=20, n_estimators=400,
                          random_state=18)
pls_reg = PLSRegression(max_iter=1000, n_components=20, scale=False)
lasso_reg = Lasso(alpha=0.0008, max_iter=10000)
ridge_reg = Ridge(alpha=9.75)
best_parameters = {'max_depth': 10, 'max_features': 'auto',
                   'max_samples': 0.85, 'min_samples_leaf': 5,
                   'n_jobs': 8, 'oob_score': True, 'random_state':18,
                   'n_estimators': 600}
rf = RandomForestRegressor(**best_parameters)

# An Average of two models
elastic_model.fit(X, np.log(y + 1))
y_test_pred1 = np.exp(elastic_model.predict(X_test))-1

gbr_model.fit(X, np.log(y + 1))
y_test_pred2 = np.exp(gbr_model.predict(X_test)) - 1
sample_subm['SalePrice'] = (y_test_pred1 + y_test_pred2)/2
sample_subm[['Id', 'SalePrice']].to_csv(_out_dir + 'ElasticNet_RGBM_Avg_V5.csv', index=False)

from sklearn.ensemble import VotingRegressor
# A voting Classifier
vot_reg = VotingRegressor(estimators=[('elastic',elastic_model), ('gbr',gbr_model)], n_jobs=-1)
vot_reg.fit(X, np.log(y + 1))
y_test_pred = np.exp(vot_reg.predict(X_test)) - 1
sample_subm['SalePrice2'] = y_test_pred #+ y_test_pred2)/2
sample_subm[['Id', 'SalePrice']].to_csv(_out_dir + 'VotingRegressor_Elastic_RGBM_V5.csv', index=False)


from sklearn.ensemble import StackingRegressor
# A stacking Regressor
estimators=[('elastic',elastic_model), ('gbr',gbr_model)]
stack_reg = StackingRegressor(estimators,
                            final_estimator= LinearRegression(),
                            n_jobs=-1)
stack_reg.fit(X, np.log(y + 1))
y_test_pred = np.exp(stack_reg.predict(X_test)) - 1
sample_subm['SalePrice'] = y_test_pred #+ y_test_pred2)/2
sample_subm[['Id', 'SalePrice']].to_csv(_out_dir + 'StackingRegressor_Elastic_RGBM_V5.csv', index=False)



from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
estimators=[('elastic',elastic_model), ('gbr',gbr_model), ('rf',rf),
            ('pls',pls_reg), ('rdg',ridge_reg), ('lasso',lasso_reg)]
stack_reg = StackingRegressor(estimators,
                            final_estimator= LinearRegression(),
                            n_jobs=-1)
# Selecting the base estimators and final estimator for stacking Regressor using cross validation
# Using simple train test split
X_train, X_val, y_train, y_val = train_test_split(  X, y, train_size=.75, random_state=42)
stack_reg.fit(X_train, np.log(y_train + 1))
y_val_pred = np.exp(stack_reg.predict(X_val)) - 1
rmsle(y_val, y_val_pred)

# using the cross validation
scores = cross_validate(stack_reg, X, np.log(y + 1), scoring='neg_root_mean_squared_error', return_train_score=True)
print(scores['train_score'])
print(scores['test_score'])
print(scores['train_score'].mean())
print(scores['test_score'].mean())

estimators=[('elastic',elastic_model), ('gbr',gbr_model)]
stack_reg = StackingRegressor(estimators,
                            final_estimator= LinearRegression(),
                            n_jobs=-1)
stack_reg.fit(X, np.log(y + 1))
y_test_pred = np.exp(stack_reg.predict(X_test)) - 1
sample_subm['SalePrice'] = y_test_pred #+ y_test_pred2)/2
sample_subm[['Id', 'SalePrice']].to_csv(_out_dir + 'StackingRegressor_Elastic_RGBM_PLS_V5.csv', index=False)




sample_subm['SalePrice'] = y_test_pred #+ y_test_pred2)/2



rmsle(sample_subm['SalePrice2'], sample_subm['SalePrice'])
rmsle(sample_subm['SalePrice3'], sample_subm['SalePrice'])
rmsle(sample_subm['SalePrice3'], sample_subm['SalePrice2'])






coefs = pd.DataFrame({'features':X.columns, 'coefs': elastic_model.coef_})

train_orig['SalePrice_pred_ElasticNet'] = np.exp(elastic_model.predict(X))-1
train_orig.to_csv(_home_dir+'analysis/train_elastic_net.csv', index=False)

sns.scatterplot(train['SalePrice'], train['SalePrice_pred'])
sns.scatterplot(train['SalePrice'], train['SalePrice'])
plt.savefig(_home_dir+'plots/ElasticNet_Preds.png')
plt.show()
