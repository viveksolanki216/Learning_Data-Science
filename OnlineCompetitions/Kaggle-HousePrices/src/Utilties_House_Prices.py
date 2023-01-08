import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def get_dir():
    _home_dir = "/home/vss/Documents/Personal/Personal Learnings - Data Science/OnlineCompetitions/Kaggle-HousePrices/"
    _in_dir = _home_dir + "input/"
    _out_dir = _home_dir + "output/"
    return _home_dir, _in_dir, _out_dir

def get_data():
    train = pd.read_csv(f'{_in_dir}train.csv')
    test = pd.read_csv(f'{_in_dir}test.csv')
    sample_subm = pd.read_csv(f'{_in_dir}sample_submission.csv')
    return train, test, sample_subm

def rmsle(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def print_numeric_super_baseline_model(y, score):
    simple_mean = y.mean().__round__()
    y_pred = np.repeat(simple_mean,y.shape[0])
    print("Simple Mean Model : ", score(y, y_pred).round(2))
    return None

def check_missing_rate_train_test(train, test):
    train_test_missing_rate = pd.concat([
        check_missing_rate(train),
        check_missing_rate(test)
    ], axis=1)
    train_test_missing_rate.columns = ['# train', '% train', '# test', '% test']
    return train_test_missing_rate


def check_missing_rate(data):
    missings_count = data.isnull().sum()
    missings = np.round(data.isnull().sum() / len(data) *100)
    missings = pd.concat([missings_count, missings], axis=1)
    missings.columns = ['#', '%']
    missings = missings.sort_values(['#'],ascending=False)
    #plt.figure(figsize=(20, 6))
    #missings[missings>0].plot.bar()
    #plt.axhline(0.5, color = 'r')
    #plt.show()
    return missings[ missings['#'] > 0 ]

def feature_engineering(data):
    print(data['FullBath'].value_counts())
    print(data.groupby('GarageCars')['GarageArea'].mean())
    data['f_GarageCars_and_Area_Avg'] = (
                                                data['GarageArea'] +
                                                data.groupby('GarageCars')['GarageArea'].transform(np.mean)
                                        ) / 2

    data['f_TotalBsmtSF_and_1stFlrSF_avg'] = (data['TotalBsmtSF'] + data['1stFlrSF']) / 2

    #data = data.drop(['TotRmsAbvGrd', 'GarageArea',
    #           'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'GarageYrBlt'],
    #          axis=1)

    # 13 Mar
    # feture engineering a new feature "TotalFS"
    data['TotalSF'] = (data['TotalBsmtSF']
                           + data['1stFlrSF']
                           + data['2ndFlrSF'])

    data['YrBltAndRemod'] = data['YearBuilt'] + data['YearRemodAdd']

    data['Total_sqr_footage'] = (data['BsmtFinSF1']
                                     + data['BsmtFinSF2']
                                     + data['1stFlrSF']
                                     + data['2ndFlrSF']
                                     )

    data['Total_Bathrooms'] = (data['FullBath']
                                   + (0.5 * data['HalfBath'])
                                   + data['BsmtFullBath']
                                   + (0.5 * data['BsmtHalfBath'])
                                   )

    data['Total_porch_sf'] = (data['OpenPorchSF']
                                  + data['3SsnPorch']
                                  + data['EnclosedPorch']
                                  + data['ScreenPorch']
                                  + data['WoodDeckSF']
                                  )
    data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    ## Zoning class are given in numerical; therefore converted to categorical variables. 
    data['MSSubClass'] = data['MSSubClass'].astype(str)
    data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    ## Important years and months that should be categorical variables not numerical. 
    # data['YearBuilt'] = data['YearBuilt'].astype(str)
    # data['YearRemodAdd'] = data['YearRemodAdd'].astype(str)
    # data['GarageYrBlt'] = data['GarageYrBlt'].astype(str)
    #data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)
    return data

def impute_missings(train, test):
    # Drop the columns, that has high missing-rate
    high_missing_rate_cols = ['PoolQC','MiscFeature','Alley', 'Fence']
    train.drop(high_missing_rate_cols, axis=1, inplace=True)
    test.drop(high_missing_rate_cols, axis=1, inplace=True)

    # Fill all those categorical variables with NONE, where they don't exists and hence missing
    # ie IF FirePlaces =0 ie there is not FirePlace at the home then its corresponding variables will be missing
    genuinely_missing_columns = [
        'FireplaceQu',
        'GarageCond', 'GarageFinish', 'GarageType', 'GarageQual',
        'BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2',
         'MasVnrType'
    ]
    train[genuinely_missing_columns] = train[genuinely_missing_columns].fillna('NONE')
    test[genuinely_missing_columns] = test[genuinely_missing_columns].fillna('NONE')
    train['GarageCars'] = train['GarageCars'].fillna(0)
    test['GarageCars'] = test['GarageCars'].fillna(0)


    # If the missing value columns correlates high with other features, we can use other features to impute it
    imputer = IterativeImputer(random_state=18)
    train[['LotArea', 'LotFrontage', 'TotalBsmtSF']] = imputer.fit_transform(train[['LotArea', 'LotFrontage', 'TotalBsmtSF']])
    test[['LotArea', 'LotFrontage', 'TotalBsmtSF']] = imputer.transform(test[['LotArea', 'LotFrontage', 'TotalBsmtSF']])

    train['Electrical'] = train['Electrical'].fillna('SBrkr')

    # For the test columns which has extra missings
    test_cols = [
        'MSZoning', 'BsmtHalfBath', 'Functional', 'BsmtFullBath',
        'Utilities', 'GarageArea', 'KitchenQual', 'Exterior2nd', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'SaleType', 'Exterior1st', 'MasVnrArea',
    ]
    categorical_features = test[test_cols].select_dtypes('object').columns.tolist()
    numerical_features = [col for col in test_cols if col not in categorical_features]

    # Impute Missings, A very Novice Handling to make sure that ML model don't throw off missing values
    train[categorical_features] = train[categorical_features].fillna('NONE')
    test[categorical_features] = test[categorical_features].fillna('NONE')
    train[numerical_features] = train[numerical_features].fillna(0)
    test[numerical_features] = test[numerical_features].fillna(0)

    return train, test


# Random Forest Utilities
def get_feature_importance_random_forest(features, feature_importances_):
    feature_imp = pd.DataFrame(
        {'features': features,
         'importance': feature_importances_}
    ).sort_values(['importance'],ascending=[False])
    feature_imp['imp_cum_sum'] = feature_imp['importance'].cumsum()
    print(feature_imp)
    return feature_imp

# Preprocessing Utilities
# Encode Categorical Utilites
def onehotencode_categorical_vars(train_data_cat, test_data_cat = None):
    ohe = OneHotEncoder()
    ohe.fit(pd.concat([train_data_cat, test_data_cat],axis=0))
    train_data_cat_enc = pd.DataFrame(
        ohe.transform(train_data_cat).toarray(),
        columns=ohe.get_feature_names()
    )
    if test_data_cat is not None:
        test_data_cat_enc = pd.DataFrame(
            ohe.transform(test_data_cat).toarray(),
            columns=ohe.get_feature_names()
        )
    else:
        test_data_cat_enc = None
    return train_data_cat_enc, test_data_cat_enc, ohe


def get_high_skewed_features(data, numerical_columns, abs_skewness_threshold = 0.9):
    '''
    :return: Returns a list of skewed variables
    '''
    feature_skewness = data[numerical_columns].apply(lambda x: skew(x)).sort_values(ascending=False)
    print(feature_skewness)
    high_skewed_features = feature_skewness[ abs(feature_skewness) > abs_skewness_threshold]
    return high_skewed_features.index.tolist()
